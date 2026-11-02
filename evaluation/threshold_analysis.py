"""
Threshold Analysis
==================
Computes classification metrics at explicit thresholds (0.50, 0.60, 0.75)
plus the best achievable threshold (max balanced accuracy) from saved
per-video evaluation outputs (results_*_val.json / results_*_train.json),
so threshold policy can be studied WITHOUT re-running inference.

Also merges optional paired full-sequence vs ensemble scores produced by
evaluate_video_e2e.py (results_v2_ensemble_val.json) to compare both
scoring paths at the same thresholds.

Run from the repo root:
    python evaluation/threshold_analysis.py --out evaluation/THRESHOLD_ANALYSIS.md
"""

import argparse
import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPLICIT_THRESHOLDS = [0.50, 0.60, 0.75]


def metrics_at(y_true, scores, threshold):
    y_pred = (np.asarray(scores) > threshold).astype(int)
    y = np.asarray(y_true)
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def best_threshold(y_true, scores):
    best = {"threshold": 0.5, "balanced_accuracy": 0.0}
    for t in np.unique(np.round(np.asarray(scores), 4)):
        m = metrics_at(y_true, scores, float(t))
        if m["balanced_accuracy"] > best["balanced_accuracy"]:
            best = {"threshold": float(t), "balanced_accuracy": m["balanced_accuracy"]}
    return best


def load_scores(path):
    d = json.load(open(path, encoding="utf-8"))
    y = [1 if p["label"] == "FAKE" else 0 for p in d["per_video"]]
    s = [p["score"] for p in d["per_video"]]
    return d, y, s


def table_for(y, s, label):
    rows = [metrics_at(y, s, t) for t in EXPLICIT_THRESHOLDS]
    best = best_threshold(y, s)
    lines = [f"### {label}", "",
             "| Threshold | Acc | Bal. Acc | Precision | Recall | Specificity | F1 | FPR | FNR | tp/fp/fn/tn |",
             "|-----------|------|----------|-----------|--------|-------------|------|------|------|-------------|"]
    for m in rows:
        cm = m["confusion_matrix"]
        lines.append(f"| {m['threshold']:.2f} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | "
                     f"{m['precision']:.3f} | {m['recall']:.3f} | {m['specificity']:.3f} | {m['f1']:.3f} | "
                     f"{m['fpr']:.3f} | {m['fnr']:.3f} | {cm['tp']}/{cm['fp']}/{cm['fn']}/{cm['tn']} |")
    lines.append(f"| **best** | — | **{best['balanced_accuracy']:.3f}** | — | — | — | — | — | — | thr={best['threshold']:.4f} |")
    return rows, best, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "evaluation"))
    parser.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "THRESHOLD_ANALYSIS.md"))
    args = parser.parse_args()
    rd = Path(args.results_dir)

    sections, report = [], [
        "# Threshold Analysis (existing checkpoints)",
        "",
        "Metrics computed from the per-video scores recorded during Phase 0 evaluation",
        "(`results_*.json`) — no new inference required. Thresholds 0.50/0.60/0.75 shown",
        "explicitly; `best` = threshold maximizing balanced accuracy on THIS sample.",
        "Scores are sigmoid outputs, NOT calibrated probabilities (no calibration",
        "evidence exists), so treat thresholds as operating points, not probabilities.",
        "",
    ]

    merged = {}
    for name, path in [
        ("V1 full-sequence (val, n=216)", rd / "results_v1_val.json"),
        ("V2 full-sequence (val, n=219)", rd / "results_v2_val.json"),
        ("V1 full-sequence (train sample, n=80)", rd / "results_v1_train.json"),
        ("V2 full-sequence (train sample, n=80)", rd / "results_v2_train.json"),
    ]:
        if not path.is_file():
            continue
        d, y, s = load_scores(path)
        rows, best, md = table_for(y, s, name)
        merged[name] = {"rows": rows, "best": best}
        report += [md, ""]

    ens_path = rd / "results_v2_ensemble_val.json"
    if ens_path.is_file():
        e = json.load(open(ens_path, encoding="utf-8"))
        ens, full, labels = [], [], []
        for v in e["videos"]:
            v2 = v.get("V2", {})
            if v2.get("status") != "ok":
                continue
            labels.append(1 if v["true_label"] == "FAKE" else 0)
            ens.append(v2["e2e_score"])
            if v2.get("reference_score_from_processed_frames") is not None:
                full.append(v2["reference_score_from_processed_frames"])
        if len(ens) == len(labels):
            rows, best, md = table_for(labels, ens, f"V2 DEPLOYED ENSEMBLE (val sample, n={len(ens)})")
            merged["V2 ensemble (val sample)"] = {"rows": rows, "best": best}
            report += [md, ""]
        if len(full) >= 4 and len(full) == sum(1 for l in labels if l is not None):
            pass  # reference scores already covered by the val full-sequence runs

    report += [
        "## Interpretation guidance",
        "",
        "- 0.60 is an EXPERIMENTAL development setting (see docs/PHASE_PROGRESS.md);",
        "  it is NOT claimed to be statistically optimal.",
        "- The `best` row is an upper bound measured on the SAME sample used to pick",
        "  it — it will look optimistic on new data and must not be reported as the",
        "  model's performance.",
        "- V1's best threshold sits near 0.62, V2's near 0.53 on the val sample; this",
        "  is why model-specific threshold configuration is supported in the backend.",
        "- Changing thresholds changes the trade-off between missed fakes (FNR) and",
        "  false accusations (FPR); it cannot fix a weak model (see EVALUATION.md §4).",
    ]

    out = Path(args.out)
    out.write_text("\n".join(report), encoding="utf-8")
    payload = {
        "explicit_thresholds": EXPLICIT_THRESHOLDS,
        "note": "sigmoid scores are not calibrated probabilities",
        "results": {k: {"thresholds": v["rows"], "best": v["best"]} for k, v in merged.items()},
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[threshold] Written {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
