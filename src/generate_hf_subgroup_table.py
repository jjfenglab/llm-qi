#!/usr/bin/env python3
"""Generate the heart-failure subgroup theme-comparison supplementary table.

The prior Lean A3 analysis of readmissions focused on heart-failure patients
while the AI pipeline covered four CMS diagnosis cohorts, so themes found
only by the AI pipeline could reflect the broader case mix rather than
discovery within the Lean population. This script recomputes the theme
distribution within the heart-failure subgroup and writes a side-by-side
comparison table (paper Supplementary Information, "Heart-Failure Subgroup
Theme Comparison").

Inputs (none are shipped with this repository):
  exp_readmission/_output/prompt_v8/default/tagged_reasons.csv
      theme assignments per factor (pipeline output)
  exp_readmission/_output/prompt_v8/default/cluster_members_gemini_clean.json
      theme slug -> topic_name (pipeline output, LLM-cleaned)
  exp_readmission/_output/prompt_v8/default/cluster_to_lean_mapping.json
      Lean A3 category -> list of theme slugs (manually curated)
  exp_readmission/data/encounter_diagnosis_cohort.csv
      encounter -> CMS cohort of the preceding index admission, from the
      hospital readmission registry; columns: encounter_id,
      index_diagnosis_cohort

Output:
  _output/tables/supp_tab_hf_subgroup.tex

Also prints the summary statistics quoted in the supplement prose (Spearman
rank correlation, top-10 overlap).

TABLE_NAMES pins the display names used in the published all-themes table so
the two tables read consistently; edit it (and CATEGORY_ORDER) if you run the
pipeline on your own data.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
EXP = BASE / "exp_readmission"
RUN = EXP / "_output/prompt_v8/default"
OUT = BASE / "_output/tables/supp_tab_hf_subgroup.tex"

INPUTS = {
    "tagged": RUN / "tagged_reasons.csv",
    "clusters": RUN / "cluster_members_gemini_clean.json",
    "lean_map": RUN / "cluster_to_lean_mapping.json",
    "cohort": EXP / "data/encounter_diagnosis_cohort.csv",
}

# Theme names as printed in the existing all-themes supplementary table, so the
# two tables read consistently.
TABLE_NAMES = {
    "unresolved_clinical_status": "Unresolved Clinical Status at Discharge",
    "heart_failure_monitoring": "Insufficient Heart Failure Post-Discharge Monitoring",
    "diuretic_regimen_monitoring": "Suboptimal Diuretic Regimen/Monitoring",
    "gdmt_optimization": "Delayed or Suboptimal GDMT Optimization",
    "hfref_therapy_monitoring": "Suboptimal HFrEF Medical Therapy \\& Monitoring",
    "home_monitoring_equipment": "Lack of Home Monitoring Equipment",
    "incomplete_infection_treatment": "Premature Discharge With Incomplete Infection Tx",
    "care_coordination_gaps": "Care Coordination \\& Transitional Care Gaps",
    "post_discharge_followup": "Lack of Early Post-Discharge Follow-Up",
    "patient_education": "Inadequate Patient Education on Warning Signs",
    "home_support_assessment": "Inadequate Home Support \\& Functional Assessment",
    "medication_access_barriers": "Medication Access \\& Adherence Barriers",
    "psychiatric_addiction_engagement": "Missed Inpatient Psychiatric \\& Addiction Engagement",
    "substance_use_cardiac": "Unmanaged Substance Use Triggering Decompensation",
    "dialysis_adherence": "Barriers to Outpatient Dialysis Adherence",
    "copd_management": "Suboptimal COPD Management \\& Action Plan",
    "dietary_nonadherence": "Dietary Non-Adherence",
    "anticoagulation_management": "Suboptimal Anticoagulation Management",
    "sleep_apnea": "Undiagnosed or Unmanaged Sleep Apnea",
    "aspiration_risk": "Unmanaged Aspiration Risk \\& Dysphagia",
    "steroid_taper": "Inadequate Steroid Taper \\& Outpatient Planning",
    "tobacco_cessation": "Continued Tobacco Use Without Cessation Intervention",
}

CATEGORY_ORDER = [
    ("Inpatient Care Gaps", "Inpatient Care Gaps"),
    ("Clinical Navigation Failures", "Clinical Navigation Failures"),
    ("Social Determinants of Health", "Social Determinants of Health"),
    ("Additional AI-Discovered", "\\textit{AI-only (not in Lean)}"),
]


def main():
    missing = [str(p) for p in INPUTS.values() if not p.exists()]
    if missing:
        sys.exit(
            "Missing input file(s):\n  " + "\n  ".join(missing) + "\n"
            "This repository ships no patient data; see the module docstring "
            "for what each input is and where it comes from."
        )

    with open(INPUTS["lean_map"]) as f:
        lean_map = json.load(f)
    with open(INPUTS["clusters"]) as f:
        clusters = json.load(f)

    topic_to_slug = {v["topic_name"]: k for k, v in clusters.items()}
    slug_to_category = {
        slug: cat for cat, slugs in lean_map.items()
        if not cat.startswith("_") for slug in slugs
    }
    assert set(topic_to_slug.values()) == set(slug_to_category) == set(TABLE_NAMES)

    cohort = {}
    with open(INPUTS["cohort"]) as f:
        for row in csv.DictReader(f):
            cohort[int(row["encounter_id"])] = row["index_diagnosis_cohort"]

    full_enc = defaultdict(set)
    hf_enc = defaultdict(set)
    all_ids, n_fail = set(), 0
    with open(INPUTS["tagged"]) as f:
        for row in csv.DictReader(f):
            eid = int(row["encounter_id"])
            all_ids.add(eid)
            if row["tagging_failed"] == "True":
                n_fail += 1
                continue
            themes = json.loads(row["theme_names"])
            for name in themes:
                slug = topic_to_slug[name]
                full_enc[slug].add(eid)
                if cohort.get(eid) == "Heart Failure":
                    hf_enc[slug].add(eid)

    n_full = len(all_ids)
    n_hf = sum(1 for c in cohort.values() if c == "Heart Failure")
    n_unmapped = sum(1 for c in cohort.values() if not c)

    slugs = sorted(TABLE_NAMES, key=lambda s: -len(full_enc[s]))
    full_counts = [len(full_enc[s]) for s in slugs]
    hf_counts = [len(hf_enc[s]) for s in slugs]

    def ranks(counts):
        return [1 + sum(1 for c in counts if c > x) for x in counts]

    full_rank = dict(zip(slugs, ranks(full_counts)))
    hf_rank = dict(zip(slugs, ranks(hf_counts)))

    rho, _ = spearmanr(full_counts, hf_counts)
    top10_full = set(slugs[:10])
    top10_hf = set(sorted(slugs, key=lambda s: -len(hf_enc[s]))[:10])
    print(f"encounters: full={n_full}, HF={n_hf}, unmapped={n_unmapped}; "
          f"factors excluded for failed tagging: {n_fail}")
    print(f"Spearman rank correlation (full vs HF prevalence): {rho:.3f}")
    print(f"top-10 overlap: {len(top10_full & top10_hf)}/10")

    lines = [
        "% Generated by src/generate_hf_subgroup_table.py -- do not edit by hand.",
        "\\begin{table}[h]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Exploratory heart-failure subgroup theme comparison for the",
        "readmission case study. Since the prior Lean A3 analysis focused on heart-failure",
        "patients and the AI pipeline covered four CMS diagnosis cohorts, this",
        "analysis recomputes theme prevalence within the",
        f"heart-failure subgroup ({n_hf} of the {n_full} scale-up encounters with",
        "tagged factors).",
        "%mapped to",
        "%CMS index-diagnosis cohorts through the hospital readmission registry; one",
        "%encounter could not be mapped).",
        "Cells give the number of encounters",
        "contributing at least one factor to the theme, the percentage of the",
        "population, and the prevalence rank within the population (ties share",
        "rank). Themes are grouped by Lean A3 category as in the full theme table.}",
        "\\label{tab:hf_subgroup}",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{6.2cm}rrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{\\textbf{All encounters}} & \\multicolumn{2}{c}{\\textbf{Heart failure}} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "\\textbf{Lean Category} & \\textbf{AI-Identified Theme} & n (\\%) & Rank & n (\\%) & Rank \\\\",
        "\\midrule",
    ]

    for ci, (cat, cat_tex) in enumerate(CATEGORY_ORDER):
        cat_slugs = [s for s in slugs if slug_to_category[s] == cat]
        for i, s in enumerate(cat_slugs):
            fe, he = len(full_enc[s]), len(hf_enc[s])
            first = (
                f"\\multirow{{{len(cat_slugs)}}}{{=}}{{{cat_tex}}}" if i == 0 else ""
            )
            lines.append(
                f" {first} & {TABLE_NAMES[s]} & "
                f"{fe} ({100 * fe / n_full:.1f}) & {full_rank[s]} & "
                f"{he} ({100 * he / n_hf:.1f}) & {hf_rank[s]} \\\\"
            )
        if ci < len(CATEGORY_ORDER) - 1:
            lines.append("\\addlinespace" if ci < 2 else "\\midrule")
        if ci == 2:
            lines[-1] = "\\midrule"
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
