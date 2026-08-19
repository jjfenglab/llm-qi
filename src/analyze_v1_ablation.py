#!/usr/bin/env python3
"""Generic-prompt (round 1) theme-coverage ablation.

The tuned extraction prompts contain example content that overlaps with the
hospital's prior manual Lean analyses, so recovering the Lean findings on
held-out encounters does not by itself demonstrate discovery. The round-1
(v1) extraction prompts predate all Lean-informed tuning, so re-tagging the
surviving v1 extraction outputs with the *published* theme tagger shows how
much of the final theme taxonomy a generic prompt already reaches. The LOS
comparison is reported in the paper's Supplementary Information
("Generic-Prompt (Round 1) Theme Coverage").

Two phases:

  tag       Re-tag the v1 extraction outputs with exactly the tagger
            configuration used for the published tables (same themes JSON,
            same prompt template, same model, temperature 0; see the
            tag_themes step in exp_*/sconscript).  Shells out to
            src/tag_themes.py, so it needs the same LLM credentials as the
            pipeline (see the README).
  analyze   Coverage of published themes and prior-Lean categories by the
            v1 run.

Expected inputs (produced by running the pipeline; none are shipped with
this repository):

  exp_los/_output/all/prompt_v1/extracted_reasons.csv       round-1 outputs
  exp_los/_output/all/prompt_v9/default/
      cluster_members_gemini_clean.json                     published themes
      tagged_reasons.csv                                    published tagging
      cluster_to_lean_mapping.json    prior-Lean category -> theme slugs

Tagged v1 outputs land under exp_*/_output/, which is gitignored: the factor
text derives from clinical notes and must not be committed.

Usage:
    python src/analyze_v1_ablation.py tag
    python src/analyze_v1_ablation.py analyze
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent

# Same tagging model as the published theme-tagging runs (exp_*/sconscript).
TAGGING_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

CONDITIONS = {
    "los": {
        "v1_extracted": BASE / "exp_los/_output/all/prompt_v1/extracted_reasons.csv",
        "v1_shim": BASE / "exp_los/_output/all/prompt_v1/extracted_reasons_shim.csv",
        "v1_tagged": BASE / "exp_los/_output/all/prompt_v1/tagged_v1.csv",
        "v1_log": BASE / "exp_los/_output/all/prompt_v1/log_theme_tagging_v1.txt",
        "themes_json": BASE / "exp_los/_output/all/prompt_v9/default/cluster_members_gemini_clean.json",
        "template": BASE / "exp_los/prompts/theme_tagging_template.txt",
        "published_tagged": BASE / "exp_los/_output/all/prompt_v9/default/tagged_reasons.csv",
        "lean_map": BASE / "exp_los/_output/all/prompt_v9/default/cluster_to_lean_mapping.json",
        # v1 scored confidence on a 0-3 Likert scale; the published run scored on
        # 0-100 in a separate pass, so no comparable confidence filter exists.
        "v1_high_conf": 3.0,
    },
}


def require(cfg, keys):
    """Exit with a pointer to the expected layout if any input is missing."""
    missing = [str(cfg[k]) for k in keys if cfg[k] is not None and not cfg[k].exists()]
    if missing:
        sys.exit(
            "Missing input file(s):\n  " + "\n  ".join(missing) + "\n"
            "This repository ships no patient data; these files are produced "
            "by running the pipeline on your own data (see the module "
            "docstring and the README's 'Input data contract')."
        )


def make_shim(cfg):
    """tag_themes.py reads df['explanation_support']; the LOS v1 schema calls it
    'explanation'.  Copy it across so the tagger sees the same two fields it saw for
    the published run."""
    if cfg["v1_shim"] is None:
        return cfg["v1_extracted"]
    df = pd.read_csv(cfg["v1_extracted"])
    assert "explanation" in df.columns, f"unexpected v1 schema: {list(df.columns)}"
    df["explanation_support"] = df["explanation"]
    df.to_csv(cfg["v1_shim"], index=False)
    return cfg["v1_shim"]


def run_tagging(name, cfg):
    require(cfg, ["v1_extracted", "themes_json", "template"])
    cmd = [
        sys.executable, "src/tag_themes.py",
        "--input-csv", str(make_shim(cfg)),
        "--themes-json", str(cfg["themes_json"]),
        "--output-csv", str(cfg["v1_tagged"]),
        "--prompt-template", str(cfg["template"]),
        "--cache-db", str(BASE / f"v1_ablation_cache_{name}.db"),
        "--log-file", str(cfg["v1_log"]),
        "--model", TAGGING_MODEL,
        "--batch-size", "10",
    ]
    print(f"[{name}] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=BASE, check=True)


def theme_sets(path):
    """encounter sets and factor counts per theme name, from a tagged_reasons CSV.

    Rows with extraction_failed are placeholders for encounters where extraction
    itself errored; they carry no factor text and are dropped from all denominators.
    """
    df = pd.read_csv(path)
    if "extraction_failed" in df.columns:
        n_placeholder = int(df["extraction_failed"].astype(bool).sum())
        df = df[~df["extraction_failed"].astype(bool)].copy()
    else:
        n_placeholder = 0
    enc, factors = defaultdict(set), defaultdict(int)
    n_untagged = 0
    for _, row in df.iterrows():
        if pd.isna(row.get("theme_names")):
            n_untagged += 1
            continue
        for name in json.loads(row["theme_names"]):
            enc[name].add(row["encounter_id"])
            factors[name] += 1
    return df, enc, factors, n_untagged, n_placeholder


def analyze(name, cfg, top_k=15):
    require(cfg, ["themes_json", "lean_map", "published_tagged", "v1_tagged"])
    clusters = json.loads(cfg["themes_json"].read_text())
    lean_map = json.loads(cfg["lean_map"].read_text())
    topic_to_slug = {v["topic_name"]: k for k, v in clusters.items()}
    slug_to_cat = {
        slug: cat for cat, slugs in lean_map.items()
        if not cat.startswith("_") for slug in slugs
    }

    pub_df, pub_enc, pub_fac, _, _ = theme_sets(cfg["published_tagged"])
    v1_df, v1_enc, v1_fac, v1_untagged, v1_placeholder = theme_sets(cfg["v1_tagged"])
    n_pub_enc = pub_df["encounter_id"].nunique()
    n_v1_enc = v1_df["encounter_id"].nunique()

    ranked = sorted(topic_to_slug, key=lambda t: -len(pub_enc[t]))

    print("=" * 100)
    print(f"{name.upper()}  |  v1: {len(v1_df)} factors / {n_v1_enc} encounters   "
          f"published: {len(pub_df)} factors / {n_pub_enc} encounters")
    print(f"v1 rows dropped as extraction-failure placeholders: {v1_placeholder}")
    print(f"v1 factors matching no theme: {v1_untagged}")
    print("=" * 100)
    print(f"{'theme':<62}{'pub enc%':>10}{'v1 factors':>12}{'v1 enc':>8}")
    for i, t in enumerate(ranked):
        mark = "  <- top%d" % top_k if i < top_k else ""
        print(f"{t[:60]:<62}{100*len(pub_enc[t])/n_pub_enc:>9.1f}%"
              f"{v1_fac[t]:>12}{len(v1_enc[t]):>8}{mark}")

    top = ranked[:top_k]
    hit_top = [t for t in top if v1_fac[t] > 0]
    hit_all = [t for t in ranked if v1_fac[t] > 0]
    print(f"\nThemes surfaced by v1: {len(hit_all)}/{len(ranked)} overall, "
          f"{len(hit_top)}/{len(top)} of the top {top_k} by published encounter prevalence")
    missed = [t for t in ranked if v1_fac[t] == 0]
    if missed:
        print("Themes NOT reached by v1:")
        for t in missed:
            print(f"  - {t}  (published prevalence {100*len(pub_enc[t])/n_pub_enc:.1f}%)")

    # Prevalence-rank comparison reported in the supplement's Generic-Prompt
    # (Round 1) Theme Coverage section.
    from scipy.stats import spearmanr

    v1_ranked = sorted(topic_to_slug, key=lambda t: -len(v1_enc[t]))
    pub_rank = {t: i + 1 for i, t in enumerate(ranked)}
    v1_rank = {t: i + 1 for i, t in enumerate(v1_ranked)}
    rho, _ = spearmanr([len(pub_enc[t]) for t in ranked],
                       [len(v1_enc[t]) for t in ranked])
    print(f"\nSpearman rank correlation, v1 vs published encounter prevalence: {rho:.2f}")
    print("Largest rank shifts (published -> v1):")
    for t in sorted(ranked, key=lambda t: -abs(pub_rank[t] - v1_rank[t]))[:5]:
        print(f"  {t[:60]:<62} {pub_rank[t]:>3} -> {v1_rank[t]:>3}")

    print("\nPrior-Lean category coverage:")
    cats = [c for c in lean_map if not c.startswith("_")]
    for cat in cats:
        slugs = lean_map[cat]
        n_hit = sum(1 for s in slugs
                    if v1_fac[clusters[s]["topic_name"]] > 0)
        n_fac = sum(v1_fac[clusters[s]["topic_name"]] for s in slugs)
        print(f"  {cat:<48} themes {n_hit}/{len(slugs)}   v1 factors {n_fac}")
    lean_only = [c for c in cats if c != "Additional AI-Discovered"]
    reached = [c for c in lean_only
               if any(v1_fac[clusters[s]["topic_name"]] > 0 for s in lean_map[c])]
    print(f"  => prior-Lean categories reached: {len(reached)}/{len(lean_only)}")

    hc = cfg["v1_high_conf"]
    if hc is not None and "confidence" in v1_df.columns:
        sub = v1_df[v1_df["confidence"] >= hc]
        hc_fac = defaultdict(int)
        for _, row in sub.iterrows():
            if pd.notna(row.get("theme_names")):
                for nm in json.loads(row["theme_names"]):
                    hc_fac[nm] += 1
        hit = [t for t in ranked if hc_fac[t] > 0]
        hit_t = [t for t in top if hc_fac[t] > 0]
        print(f"\nSensitivity, v1 confidence == {hc:g} only ({len(sub)} factors): "
              f"{len(hit)}/{len(ranked)} themes, {len(hit_t)}/{len(top)} of top {top_k}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["tag", "analyze"])
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    cfg = CONDITIONS["los"]
    if args.phase == "tag":
        run_tagging("los", cfg)
    else:
        analyze("los", cfg, top_k=args.top_k)


if __name__ == "__main__":
    main()
