#!/usr/bin/env python3
"""Agreement and reliability metrics for the expert-annotated evaluation sets.

Reads the annotation databases written by the validation UI
(exp_los/annotations.db, exp_readmission/annotations.db; see
ui/start_validation_app.py) and computes, for each case study's final
annotated iteration:
  - score distributions (expert Likert 1-5; LLM confidence, both rating-weighted
    and unique-item)
  - confusion matrix (rows = expert score, columns = binned LLM score)
  - exact / within-one agreement, MAE, weighted kappa (linear + quadratic)
  - Spearman rho and Kendall tau-b on raw confidence vs expert scores
  - Krippendorff's ordinal alpha as the multi-rater human reliability
    statistic, computed on the multiply rated factors (singly rated factors
    contribute no pairable values, so subsetting leaves alpha unchanged)
  - matched comparison on the factors rated by all reviewers (enforced;
    partially rated factors raise an error): pooled pairwise human-human
    agreement, LLM-human agreement on the same items, and their difference,
    with a JOINT item bootstrap (the same resampled items feed both sides)
  - leave-one-encounter-out sensitivity for the matched comparison
  - ICC(A,1) (two-way, absolute agreement, single measure) on the largest
    complete item x rater block, humans only, raters treated as fixed
  - full-set bootstrap CIs: the primary bootstrap resamples candidate factors
    (keeping each factor's ratings together), matching the resampling unit of
    the matched analysis; an encounter-clustered bootstrap is reported as a
    sensitivity analysis for the clustering of factors within patients. One
    common set of draws for all metrics per scheme (seed 42).

Confidence -> Likert binning mirrors bin_confidence_to_score in
src/analyze_annotations.py with the per-study cutpoints from the sconscripts
(values at a cutpoint go to the higher category):
  LOS  [60,70,80,90]:  <60->1, 60-69->2, 70-79->3, 80-89->4, >=90->5
  Readm [50,60,70,80]: <50->1, 50-59->2, 60-69->3, 70-79->4, >=80->5

Estimand note: full-set metrics are rating-weighted (each human rating counts
once, matching the numbers reported in the paper), so multiply rated items
carry more weight; an item-balanced sensitivity (rows weighted 1/m_i) is
reported alongside. The primary CI resampling unit is the candidate factor
everywhere (full set and matched analysis); encounter-clustered CIs are the
sensitivity. All analyses are conditional on the observed reviewers.

Expected inputs (not shipped with this repository; see the README's "Input
data contract"): one DuckDB database per case study with a `validations`
table holding at least encounter_id, reason_id, reviewer_id, prompt_id,
feedback_type, LLM_confidence, and annotation (1-5). The validation UI in
ui/ writes exactly this schema.

Usage: python src/agreement_metrics.py [--study {los,readmission}]
Outputs: exp_los/_output/agreement_metrics.json,
         exp_readmission/_output/agreement_metrics.json
"""

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import duckdb
import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO = Path(__file__).resolve().parent.parent
SEED = 42
N_BOOT = 5000
LIKERT = [1, 2, 3, 4, 5]
MAX_BOOT_FAILURE_RATE = 0.10

STUDIES = {
    "los": {
        "db": REPO / "exp_los/annotations.db",
        "prompt_id": "exp_los_v8",
        "bin_edges": [60, 70, 80, 90],
        "out": REPO / "exp_los/_output/agreement_metrics.json",
    },
    "readmission": {
        "db": REPO / "exp_readmission/annotations.db",
        "prompt_id": "exp_readmission_v7",
        "bin_edges": [50, 60, 70, 80],
        "out": REPO / "exp_readmission/_output/agreement_metrics.json",
    },
}


# ---------------------------------------------------------------- loading

def bin_confidence(conf, bin_edges):
    conf = np.asarray(conf, dtype=float)
    assert np.all(np.isfinite(conf)), "non-finite confidence values"
    assert np.all((conf >= 0) & (conf <= 100)), "confidence outside [0,100]"
    assert list(bin_edges) == sorted(set(bin_edges)), "edges must be increasing"
    return np.digitize(conf, bin_edges, right=False) + 1


def load(db_path, prompt_id, bin_edges):
    """Return (all annotated rows, rows that also have a confidence)."""
    if not Path(db_path).exists():
        sys.exit(
            f"Annotation database not found: {db_path}\n"
            "This repository ships no patient data. Annotation databases are "
            "produced by the validation UI (ui/start_validation_app.py) when "
            "you run the pipeline on your own data; see the README's 'Input "
            "data contract'."
        )
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        """
        SELECT encounter_id, reason_id, reviewer_id,
               LLM_confidence AS confidence, annotation
        FROM validations
        WHERE prompt_id = ? AND feedback_type = 'reason'
          AND annotation IS NOT NULL
        ORDER BY encounter_id, reason_id, reviewer_id
        """,
        [prompt_id],
    ).df()
    con.close()

    dup = df.duplicated(["encounter_id", "reason_id", "reviewer_id"])
    assert not dup.any(), (
        f"{dup.sum()} duplicate (encounter, reason, reviewer) rows -- "
        "resolve before computing agreement")
    assert df["annotation"].isin(LIKERT).all(), "annotation outside 1-5"

    conf_df = df[df["confidence"].notna()].copy()
    per_item = conf_df.groupby(["encounter_id", "reason_id"])["confidence"].nunique()
    assert (per_item == 1).all(), "confidence differs within an item"
    conf_df["llm_score"] = bin_confidence(
        conf_df["confidence"].to_numpy(), bin_edges)
    return df, conf_df


def item_table(conf_df):
    """One row per item: encounter, human ratings array, LLM score."""
    items = []
    for (enc, rid), grp in conf_df.groupby(["encounter_id", "reason_id"]):
        items.append({
            "encounter_id": enc,
            "ratings": grp["annotation"].to_numpy(),
            "llm_score": int(grp["llm_score"].iloc[0]),
            "confidence": float(grp["confidence"].iloc[0]),
        })
    return items


# ---------------------------------------------------------------- metrics

def base_metrics(a, b, weights=None):
    """exact / within-one / MAE / weighted kappas between two score arrays."""
    a, b = np.asarray(a), np.asarray(b)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    diff = np.abs(a - b)
    out = {
        "exact": float(np.average(diff == 0, weights=w)),
        "within_one": float(np.average(diff <= 1, weights=w)),
        "mae": float(np.average(diff, weights=w)),
    }
    for name, scheme in [("kappa_linear", "linear"),
                         ("kappa_quadratic", "quadratic")]:
        try:
            k = cohen_kappa_score(a, b, weights=scheme, labels=LIKERT,
                                  sample_weight=w)
            out[name] = float(k) if np.isfinite(k) else None
        except Exception:
            out[name] = None
    return out


def correlations(conf, annot):
    rho, _ = stats.spearmanr(conf, annot)
    tau, _ = stats.kendalltau(conf, annot, variant="b")
    return {"spearman_rho": float(rho), "kendall_tau_b": float(tau)}


def calibration_scalars(conf_df):
    """Bin-weighted calibration errors.

    ordinal_ece: mean |bin-implied Likert score - mean expert score| across
      confidence bins, weighted by bin size (units: Likert points).
    ece_ge3 / ece_ge4: standard ECE treating confidence/100 as the probability
      that the expert rates the factor >=3 (neutral or higher) / >=4 (agree).
    """
    n_total = len(conf_df)
    oce = e3 = e4 = 0.0
    for s, sub in conf_df.groupby("llm_score"):
        w = len(sub) / n_total
        oce += w * abs(sub["annotation"].mean() - s)
        p_hat = sub["confidence"].mean() / 100.0
        e3 += w * abs(p_hat - (sub["annotation"] >= 3).mean())
        e4 += w * abs(p_hat - (sub["annotation"] >= 4).mean())
    return {"ordinal_ece": float(oce), "ece_ge3": float(e3),
            "ece_ge4": float(e4)}


def full_metrics(conf_df, weights=None):
    out = base_metrics(conf_df["llm_score"], conf_df["annotation"], weights)
    out.update(correlations(conf_df["confidence"], conf_df["annotation"]))
    out.update(calibration_scalars(conf_df))
    return out


def hh_pairs_from_items(items):
    """Pooled human-human pairs across items (a, b arrays)."""
    a_list, b_list = [], []
    for it in items:
        y = it["ratings"]
        for i, j in combinations(range(len(y)), 2):
            a_list.append(y[i]); b_list.append(y[j])
    return np.array(a_list), np.array(b_list)


def matched_metrics(items):
    """Pooled HH, matched LH, and LH-minus-HH differences on shared items."""
    a, b = hh_pairs_from_items(items)
    # symmetrize orderings: pooled pairwise kappa with rater-agnostic margins
    hh = base_metrics(np.concatenate([a, b]), np.concatenate([b, a]))
    hh.update({k: base_metrics(a, b)[k] for k in ["exact", "within_one", "mae"]})
    hh["n_pairs"] = int(len(a))

    lh_llm = np.concatenate([np.repeat(it["llm_score"], len(it["ratings"]))
                             for it in items])
    lh_hum = np.concatenate([it["ratings"] for it in items])
    lh = base_metrics(lh_llm, lh_hum)
    lh["n"] = int(len(lh_hum))

    diff = {k: (lh[k] - hh[k]) if (lh.get(k) is not None and
                                   hh.get(k) is not None) else None
            for k in ["exact", "within_one", "mae", "kappa_quadratic"]}
    return hh, lh, diff


# ---------------------------------------------------------------- bootstrap

def summarize_reps(reps_by_metric, n_boot):
    """Percentile CIs + validity accounting; None if too many failures."""
    out = {}
    for metric, reps in reps_by_metric.items():
        valid = [r for r in reps if r is not None and np.isfinite(r)]
        entry = {"n_valid": len(valid)}
        if len(valid) >= n_boot * (1 - MAX_BOOT_FAILURE_RATE):
            lo, hi = np.percentile(valid, [2.5, 97.5])
            entry["ci95"] = [float(lo), float(hi)]
        else:
            entry["ci95"] = None
        out[metric] = entry
    return out


FULL_METRIC_KEYS = ["exact", "within_one", "mae", "kappa_linear",
                    "kappa_quadratic", "spearman_rho", "kendall_tau_b",
                    "ordinal_ece", "ece_ge3", "ece_ge4"]


def _full_boot(group_frames, n_boot, seed):
    """Resample the given groups with replacement; ONE set of draws, all
    metrics per draw."""
    rng = np.random.default_rng(seed)
    k = len(group_frames)
    reps = {m: [] for m in FULL_METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, k, size=k)
        boot = pd.concat([group_frames[i] for i in idx], ignore_index=True)
        try:
            vals = full_metrics(boot)
        except Exception:
            vals = {}
        for m in FULL_METRIC_KEYS:
            reps[m].append(vals.get(m))
    return summarize_reps(reps, n_boot)


def full_set_bootstrap_factors(conf_df, n_boot=N_BOOT, seed=SEED):
    """Primary bootstrap: resample candidate factors, keeping each factor's
    ratings together; the resampling unit matches the matched analysis."""
    groups = [g for _, g in conf_df.groupby(["encounter_id", "reason_id"])]
    return _full_boot(groups, n_boot, seed)


def full_set_bootstrap(conf_df, n_boot=N_BOOT, seed=SEED):
    """Encounter-clustered bootstrap (sensitivity analysis for the
    clustering of factors within patients)."""
    groups = [g for _, g in conf_df.groupby("encounter_id")]
    return _full_boot(groups, n_boot, seed)


def matched_bootstrap(items, n_boot=N_BOOT, seed=SEED):
    """JOINT item bootstrap: each replicate resamples items and recomputes
    HH, LH, and their difference from the same item set."""
    rng = np.random.default_rng(seed)
    n = len(items)
    sides = {"human_human": ["exact", "within_one", "mae", "kappa_quadratic"],
             "llm_human": ["exact", "within_one", "mae", "kappa_quadratic"],
             "difference": ["exact", "within_one", "mae", "kappa_quadratic"]}
    reps = {s: {m: [] for m in ms} for s, ms in sides.items()}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [items[i] for i in idx]
        try:
            hh, lh, diff = matched_metrics(sample)
        except Exception:
            hh, lh, diff = {}, {}, {}
        for m in sides["human_human"]:
            reps["human_human"][m].append(hh.get(m))
            reps["llm_human"][m].append(lh.get(m))
            reps["difference"][m].append(diff.get(m))
    return {s: summarize_reps(r, n_boot) for s, r in reps.items()}


def leave_one_encounter_out(items):
    """Range of matched point estimates when dropping each encounter."""
    encs = sorted({it["encounter_id"] for it in items})
    rows = []
    for e in encs:
        sub = [it for it in items if it["encounter_id"] != e]
        if len(sub) < 2:
            continue
        hh, lh, _ = matched_metrics(sub)
        rows.append({"dropped_encounters": 1,
                     "hh_kappa_quadratic": hh.get("kappa_quadratic"),
                     "lh_kappa_quadratic": lh.get("kappa_quadratic"),
                     "hh_within_one": hh["within_one"],
                     "lh_within_one": lh["within_one"]})
    if not rows:
        return None
    def rng_of(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return [float(min(vals)), float(max(vals))] if vals else None
    return {k: rng_of(k) for k in ["hh_kappa_quadratic", "lh_kappa_quadratic",
                                   "hh_within_one", "lh_within_one"]}


# ---------------------------------------------------------------- ICC

def icc_a1(matrix):
    """ICC(A,1): two-way, absolute agreement, single measure (McGraw-Wong).

    Conventionally written ICC(2,1) under a two-way random-rater reading; we
    treat the raters as the fixed observed set. matrix: items x raters.
    """
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    if n < 2 or k < 2:
        return None
    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ((m - grand) ** 2).sum() - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else None


def _icc_self_test():
    """Shrout & Fleiss (1979) 6x4 example: published ICC(2,1) = 0.29."""
    sf = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
          [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    val = icc_a1(sf)
    assert val is not None and abs(val - 0.29) < 0.005, f"ICC self-test: {val}"


def best_complete_block(df):
    """Largest complete item x rater block via exhaustive reviewer-subset
    search (maximize items * raters; ties favor more raters)."""
    ratings = df.pivot_table(index=["encounter_id", "reason_id"],
                             columns="reviewer_id", values="annotation",
                             aggfunc="first")
    reviewers = list(ratings.columns)
    best, best_score = None, (0, 0)
    for size in range(2, len(reviewers) + 1):
        for subset in combinations(reviewers, size):
            block = ratings[list(subset)].dropna()
            score = (block.shape[0] * size, size)
            if block.shape[0] >= 2 and score > best_score:
                best, best_score = (block, list(subset)), score
    return best


# ---------------------------------------------------------------- analysis

def krippendorff_alpha_humans(df):
    """Ordinal alpha over the multiply rated factors (humans only).

    Singly rated factors contribute no pairable values, so alpha is
    identical with or without them; we subset to make the effective
    sample explicit in the output.
    """
    ratings = df.pivot_table(index=["encounter_id", "reason_id"],
                             columns="reviewer_id", values="annotation",
                             aggfunc="first")
    multi = ratings[ratings.notna().sum(axis=1) > 1]
    mat = multi.to_numpy().T  # krippendorff expects raters x units
    return {
        "alpha": float(krippendorff.alpha(reliability_data=mat,
                                          level_of_measurement="ordinal")),
        "n_items": int(multi.shape[0]),
        "n_encounters": int(multi.index.get_level_values(0).nunique()),
    }


def _bin_test():
    got = bin_confidence([59, 60, 69, 70, 89, 90], [60, 70, 80, 90])
    assert got.tolist() == [1, 2, 2, 3, 4, 5], f"bin edge test: {got}"


def analyze(study, cfg):
    all_df, conf_df = load(cfg["db"], cfg["prompt_id"], cfg["bin_edges"])
    items = item_table(conf_df)
    uniq_conf = pd.Series([it["confidence"] for it in items])
    uniq_binned = pd.Series([it["llm_score"] for it in items])

    res = {
        "study": study,
        "prompt_id": cfg["prompt_id"],
        "bin_edges": cfg["bin_edges"],
        "seed": SEED,
        "n_boot": N_BOOT,
        "data": {
            "n_ratings_annotated": int(len(all_df)),
            "n_ratings_with_confidence": int(len(conf_df)),
            "n_encounters": int(conf_df["encounter_id"].nunique()),
            "n_items": len(items),
            "n_reviewers": int(conf_df["reviewer_id"].nunique()),
        },
        "distributions": {
            "expert_likert": {int(k): int(v) for k, v in
                              conf_df["annotation"].value_counts().sort_index().items()},
            "llm_confidence_by_rating": {int(k): int(v) for k, v in
                                         conf_df["confidence"].value_counts().sort_index().items()},
            "llm_confidence_by_item": {int(k): int(v) for k, v in
                                       uniq_conf.value_counts().sort_index().items()},
            "llm_binned_by_item": {int(k): int(v) for k, v in
                                   uniq_binned.value_counts().sort_index().items()},
        },
        "confusion_matrix_rows_expert_cols_llm": confusion_matrix(
            conf_df["annotation"], conf_df["llm_score"], labels=LIKERT).tolist(),
        "krippendorff_alpha_ordinal_humans": krippendorff_alpha_humans(all_df),
    }

    # per-bin calibration table (supports the calibration figure and metric)
    per_bin = []
    for s in LIKERT:
        sub = conf_df[conf_df["llm_score"] == s]
        per_bin.append({
            "bin_likert": s,
            "n": int(len(sub)),
            "mean_confidence": float(sub["confidence"].mean()) if len(sub) else None,
            "mean_expert_score": float(sub["annotation"].mean()) if len(sub) else None,
            "frac_ge3": float((sub["annotation"] >= 3).mean()) if len(sub) else None,
            "frac_ge4": float((sub["annotation"] >= 4).mean()) if len(sub) else None,
        })
    res["calibration_per_bin"] = per_bin

    # full set: rating-weighted (primary, matches the numbers reported in the
    # paper) plus item-balanced sensitivity (rows weighted 1/m_i)
    m_i = conf_df.groupby(["encounter_id", "reason_id"])["annotation"].transform("size")
    res["llm_human_full"] = {
        "boot_note": ("Primary CIs resample candidate factors (keeping each "
                      "factor's ratings together), matching the matched "
                      "analysis; encounter-clustered CIs are a sensitivity "
                      "analysis for within-patient clustering."),
        "rating_weighted": {**full_metrics(conf_df),
                            "boot": full_set_bootstrap_factors(conf_df),
                            "boot_encounter_clustered_sensitivity":
                                full_set_bootstrap(conf_df)},
        "item_balanced_sensitivity": base_metrics(
            conf_df["llm_score"], conf_df["annotation"],
            weights=1.0 / m_i.to_numpy()),
    }

    # matched comparison on the factors rated by all reviewers
    n_reviewers = int(conf_df["reviewer_id"].nunique())
    partial = [it for it in items if 1 < len(it["ratings"]) < n_reviewers]
    if partial:
        raise ValueError(
            "Matched comparison requires factors rated by all reviewers; "
            f"found {len(partial)} partially rated factors.")
    shared = [it for it in items if len(it["ratings"]) == n_reviewers]
    if len(shared) >= 2:
        hh, lh, diff = matched_metrics(shared)
        res["matched_shared_items"] = {
            "selection": "rated_by_all_reviewers",
            "n_reviewers_per_item": n_reviewers,
            "n_items": len(shared),
            "n_encounters": len({it["encounter_id"] for it in shared}),
            "note": ("Item-matched descriptive comparison; joint item "
                     "bootstrap conditional on the observed encounters. "
                     "Human-human kappa is a pooled pairwise weighted kappa "
                     "with symmetrized human marginals, not a two-rater "
                     "Cohen kappa."),
            "human_human": hh,
            "llm_human": lh,
            "difference_llm_minus_human": diff,
            "boot": matched_bootstrap(shared),
            "leave_one_encounter_out_ranges": leave_one_encounter_out(shared),
        }

    # ICC on the best complete block, humans only, raters fixed
    # (reviewer identities are deliberately not written to the output)
    block_info = best_complete_block(all_df)
    if block_info is not None:
        block, _raters = block_info
        res["icc"] = {
            "definition": ("ICC(A,1): two-way, absolute agreement, single "
                           "measure; raters treated as the fixed observed "
                           "set; ordinal 1-5 treated as equally spaced "
                           "(descriptive)"),
            "n_items": int(block.shape[0]),
            "n_raters": int(block.shape[1]),
            "icc_a1_humans": icc_a1(block.to_numpy()),
        }

    res["metadata"] = {
        "db_sha256": hashlib.sha256(cfg["db"].read_bytes()).hexdigest()[:16],
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
            "krippendorff": __import__("importlib.metadata", fromlist=["version"]).version("krippendorff"),
        },
    }
    return res


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Agreement metrics between LLM confidence scores and "
                    "expert ratings on the held-out evaluation sets")
    parser.add_argument(
        "--study",
        choices=list(STUDIES) + ["all"],
        default="all",
        help="Which case study to analyze (default: all)",
    )
    args = parser.parse_args()

    _icc_self_test()
    _bin_test()
    names = list(STUDIES) if args.study == "all" else [args.study]
    for study in names:
        cfg = STUDIES[study]
        res = to_jsonable(analyze(study, cfg))
        cfg["out"].parent.mkdir(parents=True, exist_ok=True)
        cfg["out"].write_text(json.dumps(res, indent=2, allow_nan=False))
        d, full = res["data"], res["llm_human_full"]["rating_weighted"]
        print(f"\n===== {study} ({res['prompt_id']}) -> {cfg['out']}")
        print(f"ratings={d['n_ratings_with_confidence']} items={d['n_items']} "
              f"encounters={d['n_encounters']} reviewers={d['n_reviewers']}")
        ka = res["krippendorff_alpha_ordinal_humans"]
        print(f"alpha(ordinal, humans)={ka['alpha']:.3f} "
              f"({ka['n_items']} multiply rated items)")
        print(f"LLM-human full: exact={full['exact']:.3f} "
              f"within1={full['within_one']:.3f} mae={full['mae']:.3f} "
              f"kq={full['kappa_quadratic']:.3f} rho={full['spearman_rho']:.3f} "
              f"tau_b={full['kendall_tau_b']:.3f}")
        for m, e in full["boot"].items():
            print(f"  {m}: ci95={e['ci95']} (n_valid={e['n_valid']})")
        ib = res["llm_human_full"]["item_balanced_sensitivity"]
        print(f"  item-balanced: exact={ib['exact']:.3f} "
              f"within1={ib['within_one']:.3f} kq={ib['kappa_quadratic']:.3f}")
        ms = res.get("matched_shared_items")
        if ms:
            print(f"matched ({ms['n_items']} items / {ms['n_encounters']} enc):")
            print(f"  HH: exact={ms['human_human']['exact']:.3f} "
                  f"within1={ms['human_human']['within_one']:.3f} "
                  f"kq={ms['human_human']['kappa_quadratic']:.3f}")
            print(f"  LH: exact={ms['llm_human']['exact']:.3f} "
                  f"within1={ms['llm_human']['within_one']:.3f} "
                  f"kq={ms['llm_human']['kappa_quadratic']:.3f}")
            dd = ms["difference_llm_minus_human"]
            db = ms["boot"]["difference"]
            for m in ["exact", "within_one", "mae", "kappa_quadratic"]:
                print(f"  diff {m}: {dd[m]:+.3f} ci95={db[m]['ci95']}")
            print(f"  LOO ranges: {ms['leave_one_encounter_out_ranges']}")
        if "icc" in res:
            i = res["icc"]
            print(f"ICC(A,1) humans, {i['n_items']}x{i['n_raters']}: "
                  f"{i['icc_a1_humans']:.3f}")


if __name__ == "__main__":
    main()
