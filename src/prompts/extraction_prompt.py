"""Prompt template for extracting readmission reasons from clinical notes."""


def create_extraction_prompt(note_text: str) -> str:
    """Create a prompt for extracting readmission reasons from a clinical note.

    Args:
        note_text: The clinical note text to analyze

    Returns:
        Formatted prompt string for LLM extraction
    """
    prompt = f"""You are a clinical quality improvement researcher using Lean methodology to analyze hospital readmission notes. Analyze the provided clinical notes to identify the reasons for readmission. Identify both clinical factors as well as operational factors, the latter characterized using the Lean wastes (DOWNTIME) framework.

Clinical Note(s):
{note_text}

Instructions:
1. Determine if the note explicitly mentions why the patient was readmitted. Extract all distinct clinical and oprtional reasons
2. Identify clinical reasons
3. Identify operational reasons: Following the Lean waste categories DOWNTIME framework (Defects, Overproduction, Waiting, Non-utilized talent, Transportation, Inventory, Motion, Extra-processing), comprehensively identify these types of factors that may likely caused or contributed to the patient's elevated readmission risk.
4. Evidence: For each identified reason, provide an EXACT quote (word-for-word) from the note. Keep quotes under 200 characters
4. Confidence: Assign a confidence level ("high", "medium", or "low") for each reason identified.
   - "high": Reasons are explicitly stated in the note
   - "medium": Reasons are implied or can be inferred from context
   - "low": Unclear or ambiguous mention of reasons

Output requirements:
* Be concise in describing the reasons.
* Use standard medical terminology when possible
* If multiple reasons/wastes are mentioned, list them all separately

Examples of clinical reasons:
- medically complex patient
- uncontrolled blood pressure
- respiratory failure
- septic shock
- acute kidney injury
- cardiac arrest

Examples of operational reasons:
- "patient did not understand discharge instructions"
- "medication non-adherence - patient never filled prescription"
- "missed follow-up appointment - no transportation arranged"
- "inadequate discharge planning for complex medical needs"
- "poor care coordination between teams"
- "communication failure regarding medication changes"
- "premature discharge without ensuring stability"
- "lack of patient education on warning signs"

Extract the readmission reasons in structured JSON format detailed below:
**IMPORTANT:** The `relevant_quotes` field must contain an EXACT quote from the clinical note - word-for-word, with no paraphrasing or summarization. Keep it under 200 characters and select the most relevant portion.

**Note:** If supporting evidence spans multiple sections of the note, you can include text from different parts separated by "..." (e.g., "patient had respiratory failure...later developed septic shock"). The validation interface will intelligently highlight each section separately to show all supporting evidence.

Example 1:
{{
  "reasons": [
     {{
       "reason": "respiratory failure",
       "confidence": "high",
       "relevant_quotes": "Readmitted to ICU for respiratory failure secondary to pneumonia and septic shock"
     }},
     {{
       "reason": "medication non-adherence due to unclear discharge instructions",
       "confidence": "medium",
       "relevant_quotes": "Patient states he did not understand when to take medications"
     }},
     {{
       "reason": "inadequate follow-up planning",
       "confidence": "medium",
       "relevant_quotes": "Had no follow-up scheduled"
     }}
}}

Example 2 (no reasons found):
{{
  "reasons": [],
}}
"""

    return prompt
