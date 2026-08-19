#!/usr/bin/env python3
"""Generate LaTeX tables for the paper's supplementary "Additional Agreement
Analyses" section from the agreement_metrics.json outputs.

Usage: python src/generate_agreement_supplement.py
Inputs: exp_los/_output/agreement_metrics.json,
        exp_readmission/_output/agreement_metrics.json
        (produced by src/agreement_metrics.py)
Output: one file per table in _output/tables/ (supp_tab_*.tex), each
\\input by the supplement next to the paragraph that discusses it.

Regenerate whenever src/agreement_metrics.py is rerun.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "_output/tables"

INPUTS = {
    "LOS": REPO / "exp_los/_output/agreement_metrics.json",
    "Readmission": REPO / "exp_readmission/_output/agreement_metrics.json",
}

METRIC_LABELS = [
    ("exact", "Exact agreement", "pct"),
    ("within_one", "Within-one agreement", "pct"),
    ("mae", "Mean absolute error (Likert points)", "num"),
    ("kappa_linear", "Weighted $\\kappa$ (linear)", "num"),
    ("kappa_quadratic", "Weighted $\\kappa$ (quadratic)", "num"),
    ("spearman_rho", "Spearman $\\rho$ (raw confidence)", "num"),
    ("kendall_tau_b", "Kendall $\\tau_b$ (raw confidence)", "num"),
    ("ordinal_ece", "Ordinal calibration error (Likert points)", "num"),
    ("ece_ge3", "ECE, $P(\\text{rating} \\geq 3)$", "num3"),
    ("ece_ge4", "ECE, $P(\\text{rating} \\geq 4)$", "num3"),
]


def fmt(v, kind, ci=None):
    if v is None:
        return "--"
    d = 3 if kind == "num3" else 2
    s = f"{100 * v:.1f}\\%" if kind == "pct" else f"{v:.{d}f}"
    if ci:
        lo, hi = ci
        if kind == "pct":
            s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
        else:
            s += f" [{lo:.{d}f}, {hi:.{d}f}]"
    return s


def fmt_range(rng, kind):
    if not rng:
        return "--"
    lo, hi = rng
    if kind == "pct":
        return f"{100 * lo:.1f}--{100 * hi:.1f}\\%"
    return f"{lo:.2f}--{hi:.2f}"


def _ece3_ucl_cell(fullm, key, kind, ci, sname):
    """LOS threshold-3 ECE point estimate is near zero: report the one-sided
    95% upper confidence limit (a two-sided percentile interval can sit
    above the point estimate here)."""
    if key == "ece_ge3" and sname == "LOS" and ci:
        return f"{fullm.get(key):.3f} (95\\% UCL {ci[1]:.3f})"
    return fmt(fullm.get(key), kind, ci)


def full_table(res):
    rows = []
    for key, label, kind in METRIC_LABELS:
        cells = []
        for sname, study in res.items():
            fullm = study["llm_human_full"]["rating_weighted"]
            ci = fullm["boot"].get(key, {}).get("ci95")
            cells.append(_ece3_ucl_cell(fullm, key, kind, ci, sname))
        rows.append(f"    {label} & {cells[0]} & {cells[1]} \\\\")
    sens_rows = []
    for key, label, kind in METRIC_LABELS[:5]:
        cells = []
        for study in res.values():
            ib = study["llm_human_full"]["item_balanced_sensitivity"]
            cells.append(fmt(ib.get(key), kind))
        sens_rows.append(f"    {label} & {cells[0]} & {cells[1]} \\\\")
    clust_rows = []
    for key, label, kind in METRIC_LABELS:
        cells = []
        for sname, study in res.items():
            fullm = study["llm_human_full"]["rating_weighted"]
            ci = fullm["boot_encounter_clustered_sensitivity"].get(key, {}).get("ci95")
            if key == "ece_ge3" and sname == "LOS" and ci:
                cells.append(f"(95\\% UCL {ci[1]:.3f})")
            elif ci is None:
                cells.append("--")
            else:
                lo, hi = ci
                if kind == "pct":
                    cells.append(f"[{100 * lo:.1f}, {100 * hi:.1f}]")
                else:
                    d = 3 if kind == "num3" else 2
                    cells.append(f"[{lo:.{d}f}, {hi:.{d}f}]")
        clust_rows.append(f"    {label} & {cells[0]} & {cells[1]} \\\\")
    counts = "; ".join(
        f"{sname if sname == 'LOS' else sname.lower()}: "
        f"{s['data']['n_ratings_with_confidence']} ratings, "
        f"{s['data']['n_encounters']} encounters, "
        f"{s['data']['n_reviewers']} reviewers"
        for sname, s in res.items()
    )
    nl = "\n"
    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{LLM-rater agreement across the full held-out evaluation sets ({counts}).
The first block weights each expert rating equally, so a factor rated by several reviewers contributes once per rating; the second block is a sensitivity analysis that instead weights each candidate factor equally, so multiply-rated factors do not contribute more than singly-rated ones.
Brackets are 95\\% bootstrap confidence intervals whose replicates resample candidate factors, keeping each factor's ratings together; the third block reports encounter-clustered intervals as a sensitivity analysis for the clustering of factors within patients.
The ordinal calibration error is the mean absolute difference, weighted by bin size, between each confidence bin's implied Likert score and the mean expert rating in that bin.
For the expected calibration errors (ECE), the confidence score is rescaled to 0-1 (confidence/100) and treated as a predicted probability that an expert rates the factor at least 3 (neutral or higher) or at least 4 (agree or higher).
The ECE is the mean absolute difference between this predicted probability and the observed fraction of these ratings in each confidence bin, weighted by the bin size.
For the LOS case study, the point estimate of the $P(\\text{{rating}} \\geq 3)$ ECE is near zero, so we report a one-sided 95\\% upper confidence limit (UCL) rather than a two-sided interval.}}
\\label{{tab:supp_agreement_full}}
\\begin{{tabular}}{{lcc}}
\\toprule
 & LOS & Readmission \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Each expert rating weighted equally}}}} \\\\
{nl.join(rows)}
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Sensitivity analysis: each candidate factor weighted equally}}}} \\\\
{nl.join(sens_rows)}
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Sensitivity analysis: encounter-clustered 95\\% CIs}}}} \\\\
{nl.join(clust_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def matched_table(res):
    blocks = [
        ("human_human", "Inter-rater (pooled pairwise)"),
        ("llm_human", "LLM-rater (same factors)"),
        ("difference", "Difference (LLM $-$ inter-rater)"),
    ]
    metrics = [
        ("exact", "Exact", "pct"),
        ("within_one", "Within one", "pct"),
        ("mae", "MAE", "num"),
        ("kappa_quadratic", "Wtd.\\ $\\kappa$ (quad.)", "num"),
    ]
    lines = []
    for bkey, blabel in blocks:
        lines.append(f"    \\multicolumn{{3}}{{l}}{{\\textit{{{blabel}}}}} \\\\")
        for mkey, mlabel, kind in metrics:
            cells = []
            for study in res.values():
                ms = study["matched_shared_items"]
                if bkey == "difference":
                    v = ms["difference_llm_minus_human"].get(mkey)
                else:
                    v = ms[bkey].get(mkey)
                ci = ms["boot"][bkey].get(mkey, {}).get("ci95")
                cells.append(fmt(v, kind, ci))
            lines.append(f"    {mlabel} & {cells[0]} & {cells[1]} \\\\")
        if bkey != "difference":
            lines.append("    \\midrule")
    # The leave-one-encounter-out block is emitted commented out (it is not
    # shown in the published table); the values are still computed and kept
    # here so the block can be restored.
    lines.append("    % \\midrule")
    lines.append(
        "    % \\multicolumn{3}{l}{\\textit{Leave-one-encounter-out ranges}} \\\\"
    )
    for mkey, mlabel, kind in [
        ("within_one", "Within one", "pct"),
        ("kappa_quadratic", "Wtd.\\ $\\kappa$ (quad.)", "num"),
    ]:
        for side, slabel in [("hh", "inter-rater"), ("lh", "LLM-rater")]:
            cells = []
            for study in res.values():
                rng = study["matched_shared_items"][
                    "leave_one_encounter_out_ranges"
                ].get(f"{side}_{mkey}")
                cells.append(fmt_range(rng, kind))
            lines.append(f"    % {mlabel}, {slabel} & {cells[0]} & {cells[1]} \\\\")
    lines.append("    \\midrule")
    lines.append(
        "    \\multicolumn{3}{l}{\\textit{Expert-rater reliability (same factors)}} \\\\"
    )
    alpha = [
        f"{fmt(s['krippendorff_alpha_ordinal_humans']['alpha'], 'num')} "
        f"({s['krippendorff_alpha_ordinal_humans']['n_items']} factors)"
        for s in res.values()
    ]
    lines.append(f"    Krippendorff's ordinal $\\alpha$ & {alpha[0]} & {alpha[1]} \\\\")
    icc = [
        f"{fmt(s['icc']['icc_a1_humans'], 'num')} ({s['icc']['n_items']}$\\times${s['icc']['n_raters']})"
        for s in res.values()
    ]
    lines.append(f"    ICC(A,1), complete block & {icc[0]} & {icc[1]} \\\\")
    n = [f"{s['matched_shared_items']['n_items']}" for s in res.values()]
    ne = [f"{s['matched_shared_items']['n_encounters']}" for s in res.values()]
    nl = "\n"
    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Comparison of inter-rater and LLM-rater agreement on the factors rated by all reviewers ({n[0]} factors from {ne[0]} encounters for LOS; {n[1]} factors from {ne[1]} encounters for readmission).
% Inter-rater agreement pools all reviewer pairs within each factor; LLM-rater agreement compares the binned AI score with each expert rating on the same factors.
Brackets are 95\\% confidence intervals from a joint item-level bootstrap, conditional on the observed encounters.
% Leave-one-encounter-out ranges give the span of point estimates as each encounter is dropped in turn.
The final block reports reliability among the expert raters on these same factors: Krippendorff's ordinal $\\alpha$ over the multiply rated factors and ICC(A,1) on the complete item $\\times$ rater block; ICC(A,1) treats the Likert categories as equally spaced and is conditional on the observed reviewer set.}}
\\label{{tab:supp_agreement_matched}}
\\begin{{tabular}}{{lcc}}
\\toprule
 & LOS & Readmission \\\\
\\midrule
{nl.join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def distribution_table(res):
    lines = []
    for name, study in res.items():
        d = study["distributions"]
        expert = [str(d["expert_likert"].get(str(k), 0)) for k in range(1, 6)]
        binned = [str(d["llm_binned_by_item"].get(str(k), 0)) for k in range(1, 6)]
        lines.append(f"    {name}, expert ratings & " + " & ".join(expert) + " \\\\")
        lines.append(
            f"    {name}, AI binned scores (unique factors) & "
            + " & ".join(binned)
            + " \\\\"
        )
    conf_lines = []
    all_levels = sorted(
        {
            int(k)
            for s in res.values()
            for k in s["distributions"]["llm_confidence_by_item"]
        }
    )
    header = " & ".join(str(v) for v in all_levels)
    for name, study in res.items():
        d = study["distributions"]["llm_confidence_by_item"]
        conf_lines.append(
            f"    {name} & "
            + " & ".join(str(d.get(str(v), 0)) for v in all_levels)
            + " \\\\"
        )
    nl = "\n"
    ncols = len(all_levels)
    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Score and confidence distributions in the held-out evaluation sets.
Top: counts of expert Likert ratings (one per rating) and of binned AI scores (one per unique candidate factor).
Bottom: counts of raw AI confidence values, one per unique candidate factor.}}
\\label{{tab:supp_distributions}}
\\begin{{tabular}}{{lccccc}}
\\toprule
 & 1 & 2 & 3 & 4 & 5 \\\\
\\midrule
{nl.join(lines)}
\\bottomrule
\\end{{tabular}}

\\vspace{{1em}}

\\begin{{tabular}}{{l{"c" * ncols}}}
\\toprule
Confidence & {header} \\\\
\\midrule
{nl.join(conf_lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def confusion_table(res):
    panels = []
    for name, study in res.items():
        cm = study["confusion_matrix_rows_expert_cols_llm"]
        rows = [
            f"    {i + 1} & " + " & ".join(str(v) for v in row) + " \\\\"
            for i, row in enumerate(cm)
        ]
        nl = "\n"
        panels.append(f"""\\begin{{minipage}}{{0.45\\textwidth}}
\\centering
{name}\\\\[0.3em]
\\begin{{tabular}}{{l|ccccc}}
\\toprule
Expert $\\backslash$ AI & 1 & 2 & 3 & 4 & 5 \\\\
\\midrule
{nl.join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}""")
    joined = "\n\\hfill\n".join(panels)
    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Confusion matrices of expert ratings (rows) versus binned AI scores (columns), counted over all ratings in the held-out evaluation sets.}}
\\label{{tab:supp_confusion}}
{joined}
\\end{{table}}"""


def calibration_table(res):
    lines = []
    for name, study in res.items():
        lines.append(f"    \\multicolumn{{6}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for row in study["calibration_per_bin"]:
            if row["n"] == 0:
                continue
            lines.append(
                f"    {row['bin_likert']} & {row['n']} & "
                f"{row['mean_confidence']:.0f} & "
                f"{row['mean_expert_score']:.2f} & "
                f"{100 * row['frac_ge3']:.0f}\\% & "
                f"{100 * row['frac_ge4']:.0f}\\% \\\\"
            )
        if name != list(res)[-1]:
            lines.append("    \\midrule")
    nl = "\n"
    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Per-bin calibration of AI confidence against expert ratings.
Each confidence bin maps to the Likert score in the first column; columns give the number of ratings in the bin, the mean raw confidence, the mean expert rating, and the fraction of expert ratings of at least 3 (neutral or higher) and at least 4 (agree or higher).}}
\\label{{tab:supp_calibration}}
\\begin{{tabular}}{{cccccc}}
\\toprule
Implied score & $n$ & Mean conf. & Mean expert & $\\geq 3$ & $\\geq 4$ \\\\
\\midrule
{nl.join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


HEADER = (
    "% Auto-generated by src/generate_agreement_supplement.py"
    " -- do not edit by hand.\n"
)

TABLES = {
    "supp_tab_agreement_full.tex": full_table,
    "supp_tab_agreement_matched.tex": matched_table,
    "supp_tab_distributions.tex": distribution_table,
    "supp_tab_confusion.tex": confusion_table,
    "supp_tab_calibration.tex": calibration_table,
}


def main():
    res = {}
    for name, path in INPUTS.items():
        if not path.exists():
            sys.exit(
                f"Missing input: {path}\n"
                "Run src/agreement_metrics.py first to produce the "
                "agreement_metrics.json files."
            )
        res[name] = json.loads(path.read_text())

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, build in TABLES.items():
        path = OUTDIR / name
        path.write_text(HEADER + build(res) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
