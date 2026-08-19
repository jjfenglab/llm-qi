#!/usr/bin/env python3
"""Generate the cohort-characteristics and study-period supplementary tables.

TRIPOD-LLM and SQUIRE ask for a description of the encounters the pipeline
was developed and evaluated on, and for the calendar window they cover
(paper Supplementary Information, "Cohort Characteristics and Study
Period"). This script assembles both from the source EHR database.

Each study has three stages, whose encounter sets are derived here from the
artifacts each stage left behind:

  refinement  encounters carrying expert ratings for any prompt before the
              evaluation prompt in exp_{los,readmission}/annotations.db
              (prompt_id != exp_los_v8 / exp_readmission_v7).
  evaluation  encounters carrying expert ratings for the evaluation prompt in
              the same databases (prompt_id exp_los_v8 / exp_readmission_v7).
  scale-up    encounters in the final tagged-factor CSVs
              (exp_*/_output/.../tagged_reasons.csv).

Refinement counts expert-rated encounters only. Each iteration round ran the
pipeline on a larger batch, but the expert rated a subset, so encounters that
were processed without being rated do not enter the table. LOS ratings
labeled exp_los_v7 count as refinement; one of those six encounters also
belongs to the 52-encounter batch rerun as the exp_los_v8 evaluation, so LOS
refinement and evaluation overlap by one.

Encounter identity differs between the studies. In the LOS study encounter_id
is the admission under review. In the readmission study it is the
*readmission* encounter; index_diagnosis_cohort in the readmissions registry
describes the preceding index admission, which is how the CMS cohort labels
are assigned.

Inputs (none are shipped with this repository):
  exp_los/annotations.db, exp_readmission/annotations.db
      validation-UI databases (see ui/start_validation_app.py)
  exp_los/_output/all/prompt_v9/default/tagged_reasons.csv
  exp_readmission/_output/prompt_v8/default/tagged_reasons.csv
      final tagged-factor CSVs (pipeline outputs)
  --ehr-db (default: the READMISSION_DB environment variable)
      DuckDB database with the source EHR tables this script reads:
        account_detail  pat_enc_csn_id_surrogate, pat_id_surrogate,
                        admission_datetime, discharge_datetime, drg_name
        adt             pat_enc_csn_id_surrogate, pat_id_surrogate, event_time
        note_meta       pat_enc_csn_id_surrogate (one row per note)
        readmissions    readmission_csn_surrogate, index_diagnosis_cohort
        demographics    pat_id_surrogate, patient_birth_date, patient_sex,
                        patient_language, patient_ethnic_group, race1, race2
  --los-diagnosis-csv (optional)
      CSV mapping encounter_id -> parent_diagnosis_code (ICD-10 parent code)
      for the LOS study cohort; without it every LOS encounter is reported
      as "Not classified".

If the dates in your source data are shifted by a uniform number of days for
de-identification, pass --date-shift-days so the study-period table reports
real calendar months; differences between two shifted dates (age at
admission, length of stay) need no adjustment.

Outputs:
  _output/tables/supp_tab_cohort_los.tex   cohort characteristics, LOS study
  _output/tables/supp_tab_cohort_readm.tex cohort characteristics, readmission
  _output/tables/supp_tab_study_period.tex calendar window per study and stage

Prints the same numbers plus join-failure and stage-overlap QC counts to
stdout.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "_output/tables"
SMALL_CELL = 5  # counts of 1..4 are reported as "<5"

STUDIES = {
    "los": {
        "label": "Length of stay",
        "annotations": BASE / "exp_los" / "annotations.db",
        "eval_prompt_id": "exp_los_v8",
        "scaleup": BASE / "exp_los/_output/all/prompt_v9/default/tagged_reasons.csv",
    },
    "readm": {
        "label": "Readmission",
        "annotations": BASE / "exp_readmission" / "annotations.db",
        "eval_prompt_id": "exp_readmission_v7",
        "scaleup": BASE / "exp_readmission/_output/prompt_v8/default/tagged_reasons.csv",
    },
}
STAGES = ["refinement", "evaluation", "scaleup"]
STAGE_LABEL = {
    "refinement": "Refinement",
    "evaluation": "Evaluation",
    "scaleup": "Scale-up",
}

LOS_DX_LABEL = {
    "A41": "Sepsis (A41)",
    "I63": "Ischemic stroke (I63)",
    "S06": "Intracranial injury (S06)",
    "T40": "Drug poisoning (T40)",
    "L03": "Cellulitis (L03)",
}
READM_DX_ORDER = ["Heart Failure", "COPD", "Pneumonia", "AMI", "Stroke"]

# Language collapse. Toishanese (Taishanese) is a Yue Chinese variety spoken in
# San Francisco's Chinatown; --toishanese-as-chinese moves it into the Chinese
# group, which is the more clinically meaningful grouping but departs from a
# literal Cantonese-plus-Mandarin definition. Off by default so the printed
# table matches the stated definition; the alternative count is always reported.
CHINESE_STRICT = {"Cantonese", "Mandarin", "Chinese"}
CHINESE_WIDE = CHINESE_STRICT | {"Toishanese"}


# --------------------------------------------------------------------------
# Stage encounter sets
# --------------------------------------------------------------------------

def _require(path, hint):
    if not Path(path).exists():
        sys.exit(
            f"Missing input: {path}\n{hint}\n"
            "This repository ships no patient data; see the module docstring "
            "for the expected inputs."
        )


def _ids_from_csv(path):
    with open(path) as fh:
        return {
            str(row["encounter_id"]).strip()
            for row in csv.DictReader(fh)
            if row.get("encounter_id") and str(row["encounter_id"]).strip()
        }


def stage_sets(study):
    cfg = STUDIES[study]
    _require(cfg["annotations"],
             "Annotation databases are written by ui/start_validation_app.py.")
    _require(cfg["scaleup"],
             "Tagged-factor CSVs are written by the tag_themes pipeline step.")
    con = duckdb.connect(str(cfg["annotations"]), read_only=True)
    refinement = {
        str(r[0]).strip()
        for r in con.execute(
            "SELECT DISTINCT encounter_id FROM validations WHERE prompt_id != ?",
            [cfg["eval_prompt_id"]],
        ).fetchall()
    }
    evaluation = {
        str(r[0]).strip()
        for r in con.execute(
            "SELECT DISTINCT encounter_id FROM validations WHERE prompt_id = ?",
            [cfg["eval_prompt_id"]],
        ).fetchall()
    }
    con.close()

    return {
        "refinement": refinement,
        "evaluation": evaluation,
        "scaleup": _ids_from_csv(cfg["scaleup"]),
    }


# --------------------------------------------------------------------------
# Source data
# --------------------------------------------------------------------------

def fetch_source(ehr_db, all_ids):
    """Pull encounter, patient, and note-count attributes for the union of stages."""
    con = duckdb.connect(ehr_db, read_only=True)
    con.execute("CREATE TEMP TABLE ids (encounter_id BIGINT)")
    con.executemany("INSERT INTO ids VALUES (?)", [(int(i),) for i in sorted(all_ids)])

    acct = con.execute("""
        SELECT a.pat_enc_csn_id_surrogate AS encounter_id, a.pat_id_surrogate,
               a.admission_datetime, a.discharge_datetime, a.drg_name
        FROM account_detail a JOIN ids i ON a.pat_enc_csn_id_surrogate = i.encounter_id
    """).df()

    # Encounters with no hospital account still have ADT movement; recover their
    # window from the first and last ADT event so they are not silently dropped.
    adt = con.execute("""
        SELECT d.pat_enc_csn_id_surrogate AS encounter_id,
               any_value(d.pat_id_surrogate) AS pat_id_surrogate,
               MIN(d.event_time) AS admission_datetime,
               MAX(d.event_time) AS discharge_datetime
        FROM adt d JOIN ids i ON d.pat_enc_csn_id_surrogate = i.encounter_id
        GROUP BY 1
    """).df()

    notes = con.execute("""
        SELECT n.pat_enc_csn_id_surrogate AS encounter_id, COUNT(*) AS n_notes
        FROM note_meta n JOIN ids i ON n.pat_enc_csn_id_surrogate = i.encounter_id
        GROUP BY 1
    """).df()

    # Readmission-study encounters are readmissions; the registry carries the
    # diagnosis cohort of the index admission that preceded them.
    registry = con.execute("""
        SELECT r.readmission_csn_surrogate AS encounter_id, r.index_diagnosis_cohort
        FROM readmissions r JOIN ids i ON r.readmission_csn_surrogate = i.encounter_id
    """).df().drop_duplicates("encounter_id")

    pat_ids = sorted(
        set(acct.pat_id_surrogate.dropna().astype("int64"))
        | set(adt.pat_id_surrogate.dropna().astype("int64"))
    )
    con.execute("CREATE TEMP TABLE pids (pat_id_surrogate BIGINT)")
    if pat_ids:  # executemany rejects an empty parameter list
        con.executemany("INSERT INTO pids VALUES (?)", [(int(p),) for p in pat_ids])
    demo = con.execute("""
        SELECT d.pat_id_surrogate, d.patient_birth_date, d.patient_sex,
               d.patient_language, d.patient_ethnic_group, d.race1, d.race2
        FROM demographics d JOIN pids p ON d.pat_id_surrogate = p.pat_id_surrogate
    """).df().drop_duplicates("pat_id_surrogate")
    con.close()

    return acct, adt, notes, registry, demo


def load_los_diagnosis(csv_path):
    """encounter_id -> ICD-10 parent code for the LOS study cohort."""
    if csv_path is None:
        print("note: --los-diagnosis-csv not given; LOS diagnosis groups will "
              "all be reported as 'Not classified'")
        return pd.DataFrame({
            "encounter_id": pd.Series(dtype="int64"),
            "parent_diagnosis_code": pd.Series(dtype=object),
        })
    dx = pd.read_csv(csv_path)[["encounter_id", "parent_diagnosis_code"]]
    dx["encounter_id"] = dx["encounter_id"].astype("int64")
    return dx.drop_duplicates("encounter_id")


def build_encounter_frame(acct, adt, notes, registry, demo, los_dx, date_shift_days):
    """One row per encounter, with the patient and stay attributes the table needs."""
    # A few encounters carry two hospital accounts spanning one continuous stay;
    # collapse to the outer window and keep whichever account named a DRG.
    acct = acct.sort_values(["encounter_id", "admission_datetime"])
    enc = acct.groupby("encounter_id").agg(
        pat_id_surrogate=("pat_id_surrogate", "first"),
        admission_datetime=("admission_datetime", "min"),
        discharge_datetime=("discharge_datetime", "max"),
        drg_name=("drg_name", "first"),
    ).reset_index()
    enc["source"] = "account_detail"

    orphans = adt[~adt.encounter_id.isin(enc.encounter_id)].copy()
    if len(orphans):
        orphans["drg_name"] = pd.NA
        orphans["source"] = "adt"
        enc = pd.concat([enc, orphans[enc.columns]], ignore_index=True)

    enc = enc.merge(demo, on="pat_id_surrogate", how="left")
    enc = enc.merge(notes, on="encounter_id", how="left")
    enc = enc.merge(registry, on="encounter_id", how="left")
    enc = enc.merge(los_dx, on="encounter_id", how="left")

    # Both sides are shifted by the same constant, so the difference is real.
    enc["age"] = (enc.admission_datetime - enc.patient_birth_date).dt.days / 365.25
    enc["los_days"] = (
        enc.discharge_datetime - enc.admission_datetime
    ).dt.total_seconds() / 86400.0
    enc["admission_real"] = enc.admission_datetime + pd.Timedelta(days=date_shift_days)
    enc["n_notes"] = enc.n_notes.fillna(0).astype(int)
    return enc.set_index("encounter_id")


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def language_group(value, wide):
    chinese = CHINESE_WIDE if wide else CHINESE_STRICT
    if pd.isna(value) or value in ("Unknown", "Declined"):
        return "Other or unknown"
    if value == "English":
        return "English"
    if value == "Spanish":
        return "Spanish"
    if value in chinese:
        return "Chinese"
    return "Other or unknown"


def race_group(row):
    if pd.notna(row.race2):
        return "More than one race"
    value = row.race1
    if pd.isna(value) or value in ("Decline to Answer", "Unknown"):
        return "Unknown or declined"
    return value


def median_iqr(values, digits=1):
    values = pd.Series(values).dropna()
    if values.empty:
        return "--"
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def count_pct(n, total):
    """Suppress counts of 1-4; a suppressed count carries no percentage."""
    if 0 < n < SMALL_CELL:
        return "$<$5"
    return f"{n} ({100.0 * n / total:.0f}\\%)" if total else "--"


def summarize(enc, ids, study, wide_chinese):
    ids = [i for i in ids if i in enc.index]
    sub = enc.loc[ids]
    n = len(sub)
    rows = {
        "N encounters": str(n),
        "Unique patients": str(sub.pat_id_surrogate.nunique()),
        "Age at admission, years": median_iqr(sub.age),
        "Length of stay, days": median_iqr(sub.los_days),
        "Notes per encounter": median_iqr(sub.n_notes, digits=0),
    }
    for sex in ["Female", "Male"]:
        rows[f"sex::{sex}"] = count_pct((sub.patient_sex == sex).sum(), n)
    other_sex = (~sub.patient_sex.isin(["Female", "Male"])).sum()
    rows["sex::Another or unknown"] = count_pct(other_sex, n)

    langs = sub.patient_language.map(lambda v: language_group(v, wide_chinese))
    for group in ["English", "Spanish", "Chinese", "Other or unknown"]:
        rows[f"lang::{group}"] = count_pct((langs == group).sum(), n)

    races = sub.apply(race_group, axis=1) if n else pd.Series(dtype=object)
    for group in [
        "White", "Black or African American", "Asian",
        "Native Hawaiian or Other Pacific Islander",
        "American Indian or Alaska Native", "Other",
        "More than one race", "Unknown or declined",
    ]:
        rows[f"race::{group}"] = count_pct((races == group).sum(), n)
    hisp = sub.patient_ethnic_group.fillna("").str.startswith("Yes")
    rows["eth::Hispanic, Latino/a, or Spanish origin"] = count_pct(hisp.sum(), n)

    if study == "los":
        for code, label in LOS_DX_LABEL.items():
            rows[f"dx::{label}"] = count_pct((sub.parent_diagnosis_code == code).sum(), n)
        rows["dx::Not classified"] = count_pct(sub.parent_diagnosis_code.isna().sum(), n)
    else:
        for label in READM_DX_ORDER:
            rows[f"dx::{label}"] = count_pct((sub.index_diagnosis_cohort == label).sum(), n)
        rows["dx::Not classified"] = count_pct(sub.index_diagnosis_cohort.isna().sum(), n)
    return rows


def study_period(enc, ids):
    ids = [i for i in ids if i in enc.index]
    dates = enc.loc[ids, "admission_real"].dropna()
    if dates.empty:
        return "--", "--"
    return dates.min().strftime("%b %Y"), dates.max().strftime("%b %Y")


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

SECTIONS = [
    (None, ["N encounters", "Unique patients", "Age at admission, years"]),
    ("Sex", ["sex::Female", "sex::Male", "sex::Another or unknown"]),
    ("Preferred language", ["lang::English", "lang::Spanish", "lang::Chinese",
                            "lang::Other or unknown"]),
    ("Race", ["race::White", "race::Black or African American", "race::Asian",
              "race::Native Hawaiian or Other Pacific Islander",
              "race::American Indian or Alaska Native", "race::Other",
              "race::More than one race", "race::Unknown or declined"]),
    ("Ethnicity", ["eth::Hispanic, Latino/a, or Spanish origin"]),
    ("Diagnosis group", None),  # filled per study
    (None, ["Length of stay, days", "Notes per encounter"]),
]


def latex_escape(text):
    return text.replace("&", "\\&").replace("%", "\\%")


def write_cohort_tex(path, summaries, study, dx_keys):
    lines = [" & " + " & ".join(STAGE_LABEL[s] for s in STAGES) + " \\\\", "\\midrule"]
    for heading, keys in SECTIONS:
        if heading == "Diagnosis group":
            keys = dx_keys
        if heading:
            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{heading}}}}} \\\\")
        for key in keys:
            label = key.split("::", 1)[-1]
            indent = "\\quad " if "::" in key else ""
            cells = " & ".join(summaries[s][key] for s in STAGES)
            lines.append(f"{indent}{latex_escape(label)} & {cells} \\\\")

    dx_note = (
        "Diagnosis groups are the ICD-10 parent codes defining the cohort."
        if study == "los" else
        "Diagnosis groups are the CMS cohort of the index admission preceding "
        "each readmission."
    )
    caption = (
        f"{STUDIES[study]['label']} study: encounters used at each stage. "
        "Median [IQR] or n (\\%); counts of one to four are suppressed as $<$5. "
        f"{dx_note}"
    )
    path.write_text(
        "% Auto-generated by src/generate_cohort_table.py -- do not edit by hand.\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:supp_cohort_{study}}}\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def write_period_tex(path, periods):
    rows = []
    for study in ["los", "readm"]:
        rows.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{STUDIES[study]['label']}}}}} \\\\")
        for stage in STAGES:
            first, last = periods[study][stage]
            rows.append(f"\\quad {STAGE_LABEL[stage]} & {first} & {last} \\\\")
    caption = (
        "Calendar window of the encounters used at each stage. Admissions are "
        "dated by the encounter under review: the index admission for the "
        "length-of-stay study, the readmission itself for the readmission study."
    )
    path.write_text(
        "% Auto-generated by src/generate_cohort_table.py -- do not edit by hand.\n"
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:supp_study_period}\n"
        "\\begin{tabular}{lcc}\n\\toprule\n"
        " & Earliest admission & Latest admission \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Cohort-characteristics and study-period tables")
    ap.add_argument("--ehr-db",
                    default=os.environ.get("READMISSION_DB"),
                    help="Path to the source EHR DuckDB database (default: "
                         "the READMISSION_DB environment variable)")
    ap.add_argument("--los-diagnosis-csv",
                    help="Optional CSV mapping encounter_id -> "
                         "parent_diagnosis_code for the LOS study cohort")
    ap.add_argument("--date-shift-days", type=int, default=0,
                    help="Uniform de-identification date shift to add back "
                         "when reporting calendar windows (default: 0)")
    ap.add_argument("--toishanese-as-chinese", action="store_true",
                    help="group Toishanese with Cantonese/Mandarin rather than Other")
    args = ap.parse_args()

    if not args.ehr_db:
        sys.exit("Set --ehr-db (or the READMISSION_DB environment variable) "
                 "to your source EHR database; see the module docstring for "
                 "the tables and columns this script reads.")
    _require(args.ehr_db, "Source EHR database not found at this path.")

    sets = {s: stage_sets(s) for s in STUDIES}
    all_ids = {i for study in sets.values() for ids in study.values() for i in ids}
    print(f"encounters across all stages: {len(all_ids)}")

    acct, adt, notes, registry, demo = fetch_source(args.ehr_db, all_ids)
    enc = build_encounter_frame(acct, adt, notes, registry, demo,
                                load_los_diagnosis(args.los_diagnosis_csv),
                                args.date_shift_days)

    print("\n--- QC: join coverage ---")
    for study, stages in sets.items():
        for stage, ids in stages.items():
            ids_int = {int(i) for i in ids}
            missing = ids_int - set(enc.index)
            no_dx = sum(
                1 for i in ids_int & set(enc.index)
                if pd.isna(enc.loc[i, "parent_diagnosis_code" if study == "los"
                                   else "index_diagnosis_cohort"])
            )
            print(f"{study:6s} {stage:11s} N={len(ids_int):4d} "
                  f"missing_encounter={len(missing)} missing_diagnosis={no_dx}")

    print("\n--- QC: stage overlap (encounters) ---")
    for study, stages in sets.items():
        for a, b in [("refinement", "evaluation"), ("refinement", "scaleup"),
                     ("evaluation", "scaleup")]:
            print(f"{study:6s} {a} n {b}: {len(stages[a] & stages[b])}")

    per_study, periods, dx_keys = {}, {}, {}
    for study, stages in sets.items():
        ids = {s: {int(i) for i in v} for s, v in stages.items()}
        per_study[study] = {
            s: summarize(enc, ids[s], study, args.toishanese_as_chinese) for s in STAGES
        }
        periods[study] = {s: study_period(enc, ids[s]) for s in STAGES}
        labels = (list(LOS_DX_LABEL.values()) if study == "los" else READM_DX_ORDER)
        dx_keys[study] = [f"dx::{l}" for l in labels] + ["dx::Not classified"]

    OUT.mkdir(parents=True, exist_ok=True)
    for study in STUDIES:
        write_cohort_tex(OUT / f"supp_tab_cohort_{study}.tex",
                         per_study[study], study, dx_keys[study])
    write_period_tex(OUT / "supp_tab_study_period.tex", periods)

    print("\n--- study period (calendar dates) ---")
    for study in STUDIES:
        for stage in STAGES:
            first, last = periods[study][stage]
            print(f"{study:6s} {stage:11s} {first} -- {last}")

    print("\n--- cohort table ---")
    for study in STUDIES:
        print(f"\n[{STUDIES[study]['label']}]")
        for key in per_study[study]["refinement"]:
            cells = "  ".join(f"{per_study[study][s][key]:>22s}" for s in STAGES)
            print(f"  {key:52s} {cells}")

    n_toi = (enc.patient_language == "Toishanese").sum()
    print(f"\nToishanese speakers in the union of stages: {n_toi} "
          f"(grouped with {'Chinese' if args.toishanese_as_chinese else 'Other or unknown'})")
    for study in STUDIES:
        print(f"\nwrote {OUT / f'supp_tab_cohort_{study}.tex'}")
    print(f"wrote {OUT / 'supp_tab_study_period.tex'}")


if __name__ == "__main__":
    main()
