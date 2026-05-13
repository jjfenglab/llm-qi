# Release Prep — Triage Notes

Audit + cleanup of `~/llm_qi_code` for the npj Digital Medicine release. Branch `release-prep` off `main` @ `c55c919`. **No commits yet** — review `git diff` before pushing.

## 1. Findings

**PHI / quasi-identifiers** (all addressed in §2)
- ~50 real ZSFG surrogate encounter IDs in `exp_readmission/sconscript:400`; 5 more in `exp_los/sconscript:38–53`.
- Reviewer first names in commented configs and example shell commands.
- Clinical excerpts + iteration narratives in `exp_los/prompts/summary.md` and `exp_readmission/prompts/summary.md`.
- **Three cluster-member JSONs under `_output/`** — re-audited by an independent agent after we considered keeping them. Found 4 calendar dates more specific than year (HIPAA), specific ZSFG-internal subunits (Ward 93, Hummingbird Place, Hummingbird Valencia), and named external facilities (Kaiser, CPMC, LHH, BAART, UCSF). These function as quasi-identifiers when combined with day-resolution timing. Per Patrick's call, **deleted** — they're regeneratable on any user's own cohort via `cluster_reasons.py` + the cleanup prompt now in `README.md`.
- `database_schema.txt` — **removed** out of caution before the repo split; the SQL in `src/assemble_readmission_notes.py` is the authoritative reference for the expected schema.

**Secrets / internal paths** (all addressed in §2)
- `requirements.txt:52` editable path under `/mnt/efs/...`.
- `ui/README_validation_app.md:72` `cd /mnt/efs/...`.
- `exp_readmission/sconscript:16` hardcoded `../zsfg_database_20250429.db`.
- `src/common.py:105–106` Versa-only assert.
- No hardcoded keys, no Wynton paths.

**Model description vs. paper** — `README.md` and `src/README.md` rewritten to match the paper (Claude Opus 4.5 default; GPT-5 / Versa / Bedrock documented as alternatives).

## 2. Changes made

**Added**:
- `LICENSE` — canonical GPL-3.0-or-later text (fetched from `https://www.gnu.org/licenses/gpl-3.0.txt`). Consistent with vendored `bloatectomy/` and now explicit for the combined work.
- `requirements-clustering.txt` — optional extras for BERTopic + theme tagging (`bertopic`, `transformers`, `sentence-transformers`, `hdbscan`, etc.). GPU wheels (`torch`, `triton`, `cuda-bindings`, `nvidia-*`) are gated to `sys_platform == "linux"`.
- `RELEASE_PREP.md` — this file.

**Deleted** (8 files):
- Both `exp_*/prompts/summary.md` (iteration history → paper Supplementary).
- All three `exp_*/_output/.../cluster_members_*.json` (quasi-identifier risk; regeneratable via the script).

**Modified**:
- `requirements.txt` — switched line 52 to `lab_llm @ git+https://github.com/jjfenglab/llm-api.git`; pulled heavy clustering + GPU deps out into `requirements-clustering.txt`; core install is now Mac/Linux/CPU friendly.
- `exp_readmission/sconscript` — hardcoded DB path → `os.environ["READMISSION_DB"]` + early assertion; reviewer first names + encounter IDs scrubbed from commented historical entries; line-400 `--exclude … --fix …` block replaced with a `<reviewer>` / `v<N>` example; cluster-cleanup docstring → short pointer.
- `exp_los/sconscript` — 5 encounter IDs removed from commented historical entries; `_dana_` CSV-filename example → `<reviewer>` placeholder; cluster-cleanup docstring → short pointer.
- `src/common.py`, `src/extract_reasons.py`, `src/extract_gantt_chart.py`, `src/score_confidences.py`, `src/tag_themes.py`, `src/filter_notes.py` — removed the 6 `sys.path.insert(..., "llm-api")` hacks (lab_llm is now a pip-installed package and resolves normally; the vendored `bloatectomy/` sys.path insert is preserved); scrubbed "Versa LLM endpoint" from docstrings; loosened the Versa-only env-var assertion to a docstring describing OpenAI / Bedrock paths.
- `ui/README_validation_app.md` — `/mnt/efs/...` → `<repo-root>`.
- `README.md` — rewritten to match the paper (citation block, repo↔paper map, both case-study walkthroughs, input contract, BERTopic cleanup prompt, Limitations); install section documents the optional clustering extras; credentials block switched from `VERSA_API_KEY` / `VERSA_ENDPOINT` to `OPENAI_ACCESS_TOKEN` + `BEDROCK_ACCESS_KEY` / `BEDROCK_ACCESS_KEY_SECRET` (with a callout that lab_llm uses the non-standard `OPENAI_ACCESS_TOKEN` name); explicit note that the cleaned cluster JSONs are not shipped (PHI scrub finding) and are regenerable.
- `src/README.md` — replaced stale GPT-4o-mini docs with a short module index; env-var section updated to OpenAI / Bedrock.

**Files explicitly left alone**: every prompt template and every other Python module in `src/`.

**Post-edit verification**: re-ran the audit greps — no `/mnt/efs`, no `zsfg_database`, no `/wynton`, no 7–9-digit encounter IDs (only `1000000` max_char), and no reviewer first names anywhere in tracked code or markdown. Independent PHI re-scan of the three cluster JSONs surfaced enough quasi-identifiers (dates + named facilities) that they were removed entirely rather than scrubbed.

## 3. Resolved + open

**Resolved:**
- **OpenAI env-var alias.** Shim added in `src/common.py:setup_llm_api` — outside teams can set `OPENAI_API_KEY` (the standard name); lab_llm's native `OPENAI_ACCESS_TOKEN` also still works.
- **`.env.example` committed** with `OPENAI_API_KEY`, `BEDROCK_*`, `READMISSION_DB`.
- **GitHub posture.** Public-now at `jjfenglab/llm-qi` with a `v0.1.0-preprint` (or similar) tag. Flip to a versioned release on acceptance for the Zenodo DOI.
- **Final PHI re-scan** (background agent, full release tree minus `bloatectomy/`/`LICENSE`/`RELEASE_PREP.md`/`do_inference.ipynb`) came back clean across HIPAA identifiers, encounter IDs, hostnames, secrets, clinical excerpts, named external facilities, and provider names. The only finding was reviewer first names in commented historical-config blocks in both sconscripts — **kept per Patrick's call** (the names are co-author and acknowledged-contributor first names, and these are commented-out, non-functional historical breadcrumbs).

**Resolved (this turn):**
- `main.tex` (theorem-proof scratch) — deleted by Patrick.
- `ui/README.md` (stale `generate_validation_page.py` docs) — `git rm`'d; canonical UI doc remains at `ui/README_validation_app.md`.
- `CITATION.cff` — deferred to acceptance.
- Synthetic fixture — skipped for v0.1.0-preprint; outside teams supply their own EHR inputs (input contract documented in `README.md`).

**Nothing material is open.** Ready to split to `jjfenglab/llm-qi`.

## 4. Repro test
- Branch off `main` @ `c55c919`; `git status` shows the expected 8 deletions + 7 modifications + 3 untracked (`LICENSE`, `requirements-clustering.txt`, `RELEASE_PREP.md`).
- **`pip install -r requirements.txt` not executed** on this macOS box. The line-52 fix is mechanically valid PEP 508 (`lab_llm @ git+https://...`); `~/llm-api/pyproject.toml` declares the matching dependency set. The clustering extras file uses standard `; sys_platform == "linux"` markers that pip will respect.
- **Pipeline end-to-end not executed.** Still requires live Versa or Bedrock credentials plus the user's own EHR inputs.
- **PHI re-scan**: independent agent reviewed all three `cluster_members_*.json` files in full and flagged 4 calendar dates + named ZSFG-internal/external facilities. Files removed.
