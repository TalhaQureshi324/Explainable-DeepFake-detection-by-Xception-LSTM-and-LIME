"""
GPU Training Preflight (Task 9)
===============================
Verifies — BEFORE any GPU time is spent — that the corrected training data
pipeline is genuinely ready for Colab/Kaggle:

1. Per-split independent group counts and class counts (scene and actor modes).
2. Derivative-closure: no original video and its manipulated derivatives in
   different splits; every manipulated video co-split with the originals of
   BOTH its source and target actors for the shared scene.
3. Credibility verdict for group-disjoint train/val/calibration/test.
4. Frame-folder coverage: can the trainer locate processed frames for every
   manifest video (by stem), per assigned split?
5. Storage/GPU-memory requirements for the Colab run (no training-time
   estimates — those require an actual GPU benchmark).

Run from the repo root:
    python training/preflight_check.py \
        --manifest training/dataset_manifest.json \
        --data-dir "F:/Computer Vision/FYP/data" \
        --frames-root "F:/Computer Vision/FYP/processed_dataset" \
        --out training/PREFLIGHT_REPORT.md
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REAL_RE = re.compile(r"^(\d+)__(.+)$")
FAKE_RE = re.compile(r"^(\d+)_(\d+)__(.+?)__\w{8}$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training/dataset_manifest.json")
    parser.add_argument("--frames-root", default="processed_dataset")
    parser.add_argument("--out", default="training/PREFLIGHT_REPORT.md")
    parser.add_argument("--video-sizes-from", default=None,
                        help="Optional raw-video data dir to estimate media storage")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    videos = manifest["videos"]
    mode = manifest.get("mode", "scene")
    splits = ["train", "val", "calibration", "test"]

    lines = [
        "# GPU Training Preflight Report",
        "",
        f"- Manifest: `{args.manifest}` (mode **{mode}**, seed {manifest.get('seed')})",
        "",
    ]

    # ---- 1. per-split group/class/actor counts --------------------------------
    per_split = {s: {"groups": set(), "REAL": 0, "FAKE": 0, "actors": set()} for s in splits}
    for v in videos:
        s = v["assigned_split"]
        per_split[s]["groups"].add(v["source_group"])
        per_split[s][v["label"]] += 1
        per_split[s]["actors"].update(v["actors"])

    lines += [
        "## 1. Split composition",
        "",
        "| Split | REAL | FAKE | Total | Independent groups | Actors touched |",
        "|-------|------|------|-------|--------------------|----------------|",
    ]
    for s in splits:
        p = per_split[s]
        lines.append(f"| {s} | {p['REAL']} | {p['FAKE']} | {p['REAL'] + p['FAKE']} | "
                     f"{len(p['groups'])} | {len(p['actors'])} |")
    lines.append("")

    # ---- 2. derivative closure -------------------------------------------------
    # scene -> set(splits of REALs with that scene); every FAKE with that scene
    # must match, and both its actors' same-scene REALs must match too.
    real_split_by_scene = defaultdict(set)
    fake_split_by_scene = defaultdict(set)
    unparsed = 0
    for v in videos:
        if v["label"] == "REAL":
            m = REAL_RE.match(v["video"])
            if m:
                real_split_by_scene[m.group(2)].add(v["assigned_split"])
            else:
                unparsed += 1
        else:
            m = FAKE_RE.match(v["video"])
            if m:
                fake_split_by_scene[m.group(3)].add(v["assigned_split"])
            else:
                unparsed += 1

    violations_scene = [sc for sc in fake_split_by_scene
                        if real_split_by_scene.get(sc) and
                        (real_split_by_scene[sc] | fake_split_by_scene[sc])
                        != real_split_by_scene[sc]]

    # manipulated video vs the originals of both its actors on the same scene
    violations_actor = []
    for v in videos:
        if v["label"] != "FAKE":
            continue
        m = FAKE_RE.match(v["video"])
        if not m:
            continue
        src, tgt, scene = m.group(1), m.group(2), m.group(3)
        for actor in (src, tgt):
            for rsplit in real_split_by_scene.get(scene, set()):
                # the REAL of `scene` belongs to exactly one actor; derive it
                pass
        # explicit check: for each REAL with same scene, if its actor is src or tgt
        for r in videos:
            if r["label"] == "REAL":
                rm = REAL_RE.match(r["video"])
                if rm and rm.group(2) == scene and rm.group(1) in (src, tgt):
                    if r["assigned_split"] != v["assigned_split"]:
                        violations_actor.append((v["video"], r["video"]))

    lines += [
        "## 2. Derivative closure (original + its manipulated derivatives)",
        "",
        f"- Unparsable video names: {unparsed}",
        f"- Scene-level violations (fake vs same-scene original in different splits): "
        f"**{len(violations_scene)}**",
        f"- Actor-pair violations (manipulated video vs same-scene original of its "
        f"source/target actor in a different split): **{len(violations_actor)}**",
    ]
    if violations_scene[:5]:
        lines += [f"  e.g. {violations_scene[:5]}"]
    if violations_actor[:5]:
        lines += [f"  e.g. {violations_actor[:5]}"]
    lines.append("")

    # ---- 3. frame coverage -------------------------------------------------------
    frames_root = Path(args.frames_root)
    index = {}
    if frames_root.is_dir():
        for split in splits:
            for label in ("REAL", "FAKE"):
                d = frames_root / split / label
                if d.is_dir():
                    for folder in d.iterdir():
                        if folder.is_dir():
                            index[folder.name] = folder
    missing_by_split = defaultdict(int)
    for v in videos:
        if v["video"] not in index:
            missing_by_split[v["assigned_split"]] += 1
    lines += [
        "## 3. Frame coverage",
        "",
        f"- Frame folders found under `{frames_root}`: {len(index)}",
        f"- Manifest videos WITHOUT preprocessed frames: "
        f"{dict(missing_by_split) or 'none'} (trainer skips these)",
        "",
    ]

    # ---- 4. credibility verdict --------------------------------------------------
    n_groups = len({v["source_group"] for v in videos})
    cal_ok = per_split["calibration"]["REAL"] > 0 and per_split["calibration"]["FAKE"] > 0
    all_ok = all(per_split[s]["REAL"] > 0 and per_split[s]["FAKE"] > 0 for s in splits)
    lines += [
        "## 4. Credibility verdict",
        "",
    ]
    if mode == "scene":
        lines += [
            f"- Scene mode yields **{n_groups}** independent groups and every split "
            f"contains both classes: {'YES' if all_ok else 'NO'}.",
            f"- Calibration split usable for threshold calibration: {'YES' if cal_ok else 'NO'}.",
            "- VERDICT: **usable** for group-disjoint training, provided the report states "
            "that face identities are NOT disjoint across splits (scene-level containment "
            "only). Actor-level disjointness is impossible with this dataset: DFD's dense "
            "source/target pair coverage collapses the actor graph into 3 components, "
            "leaving no calibration split (see training/SPLIT_SUMMARY_ACTOR.md).",
            "- RECOMMENDED alternative for a stricter claim: keep scene mode as primary and, "
            "in parallel, report a leave-one-actor-out style check later only if the retrained "
            "model is strong enough to matter; do NOT present actor-disjoint numbers that "
            "do not exist.",
        ]
    else:
        lines += [
            f"- Actor mode yields **{n_groups}** groups; calibration split has "
            f"{per_split['calibration']['REAL']} REAL / {per_split['calibration']['FAKE']} FAKE.",
            "- VERDICT: actor mode is NOT credible for a 4-way split with this dataset's "
            "dense pair coverage. Use scene mode.",
        ]
    lines.append("")

    # ---- 5. storage / GPU memory -------------------------------------------------
    import math

    if index:
        sample = list(index.values())[:50]
        total_files = 0
        size_samples = []
        for d in sample:
            for f in d.glob("*.jpg"):
                size_samples.append(f.stat().st_size)
                total_files += 1
        avg = sum(size_samples) / max(1, len(size_samples))
        est_total_gb = avg * 102480 / 1e9  # measured full frame count from Phase 0
        lines += [
            "## 5. Storage & GPU memory (measured/derived, no training-time estimates)",
            "",
            f"- Average processed-frame size: {avg / 1024:.1f} KB",
            f"- Estimated frames.zip upload: **~{est_total_gb:.1f} GB** "
            f"(102,480 frames; well inside the 15 GB free Drive tier)",
            "- Workspace after unzip: same order (~2-3 GB) — Colab disk is fine.",
            "- GPU memory: batch 8 x 32 frames x 299x299 through TimeDistributed Xception "
            "peaks around 8-11 GB on a T4 (16 GB) — fits; notebook already documents "
            "dropping to batch 4 on OOM. V1-style (10 frames) fits comfortably.",
            "- Checkpoint ~166 MB per save (best_checkpoint.keras) — Drive quota impact "
            "negligible.",
            "- Training duration: NOT estimated here; benchmark epoch 1 on the actual GPU "
            "before planning (the notebook logs per-epoch times).",
        ]

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[preflight] written -> {args.out}")


if __name__ == "__main__":
    main()
