"""Common utilities for LLM-based note processing scripts.

This module contains shared functions used by filter_notes.py and extract_reasons.py.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add vendored bloatectomy to path (the only sibling-checkout import; lab_llm
# is a pip-installed package and resolves normally).
sys.path.insert(0, str(Path(__file__).parent.parent / "bloatectomy"))
from bloatectomy import bloatectomy

from lab_llm import LLMApi, LLMCache, DuckDBHandler, ErrorCallbackHandler, ErrorTracker
from lab_llm.constants import (
    LLMModel,
    list_available_models,
    parse_model_string,
)


def deduplicate_note(note_text: str) -> str:
    """Remove duplicate text from clinical notes using bloatectomy.

    Args:
        note_text: The clinical note text to deduplicate

    Returns:
        Deduplicated note text
    """
    import re

    assert note_text is not None, "note_text cannot be None"
    assert isinstance(note_text, str), f"note_text must be string, got {type(note_text)}"
    note_text = note_text.replace('    ', '\n')

    if not note_text.strip():
        return note_text

    result = bloatectomy(note_text, style='remov', output='str')

    # Post-process: ensure newlines before timestamps
    fixed_text = re.sub(r'(?<!\n)(\d{2}:\d{2} - )', r'\n\1', result.deduplicated_string)
    return fixed_text


def load_prompt_template(template_path: str) -> str:
    """Load prompt template from file.

    Args:
        template_path: Path to the prompt template file

    Returns:
        Template string with placeholder for note text
    """
    assert Path(template_path).exists(), f"Prompt template file not found: {template_path}"

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read().strip()

    assert "{note_text}" in template, "Prompt template must contain {note_text} placeholder"

    return template


def create_prompt_from_template(template: str, note_text: str) -> str:
    """Create a prompt from template by substituting note text.

    Args:
        template: Prompt template with {note_text} placeholder
        note_text: The clinical note text to analyze

    Returns:
        Formatted prompt string for LLM extraction
    """
    return template.format(note_text=note_text)


def setup_llm_api(
    cache_db_path: str,
    log_file_path: str,
    model_str: str = "gpt-4o-mini-2024-07-18",
    timeout: int = 120,
    seed: int = 42,
) -> LLMApi:
    """Setup the lab_llm API with caching.

    Provider credentials are read from the environment by lab_llm itself based
    on the model string: OPENAI_ACCESS_TOKEN for OpenAI models (gpt-*),
    BEDROCK_ACCESS_KEY / BEDROCK_ACCESS_KEY_SECRET for Bedrock (claude-*).
    Missing credentials surface as a clear error at first call time.

    Args:
        cache_db_path: Path to cache database
        log_file_path: Path to log file (txt format)
        model_str: Model string (e.g., 'gpt-4o-2024-08-06', 'us.anthropic.claude-opus-4-5-20251101-v1:0')
        seed: Random seed for reproducibility

    Returns:
        Configured LLMApi instance
    """
    load_dotenv()

    # Accept the standard OPENAI_API_KEY env var as an alias for the
    # lab_llm-native OPENAI_ACCESS_TOKEN.
    os.environ["OPENAI_ACCESS_TOKEN"] = os.environ.get(
        "OPENAI_API_KEY", os.environ.get("OPENAI_ACCESS_TOKEN", "")
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Setting up cache: {cache_db_path}")
    cache_db = DuckDBHandler(cache_db_path, read_only=False)
    cache = LLMCache(cache_db)

    error_log_path = log_file_path.replace('.txt', '_errors.jsonl')
    logger.info(f"Setting up error tracking: {error_log_path}")
    error_tracker = ErrorTracker(error_log_path)
    error_handler = ErrorCallbackHandler(
        logger,
        error_tracker=error_tracker,
        propagate_interrupts=True
    )

    model = parse_model_string(model_str)
    logger.info(f"Using model: {model.name}")

    api = LLMApi(
        cache=cache,
        seed=seed,
        model_type=model,
        error_handler=error_handler,
        logging=logger,
        timeout=timeout,
        return_exceptions=True,
        verbosity="low"
    )

    return api
