"""
Identity/source-disjoint dataset splitting (Task R1)
====================================================

The Phase 0 audit found that the original split (script.py) assigned videos
randomly, so the same actor — and the original/manipulated pairs derived from
the same source video — appear in both train and val. This script rebuilds the
split so that:

1. A manipulated video and the original video it was derived from never land
   in different splits (related-source leakage prevented).
2. Videos sharing any actor (face identity OR scene identity) stay in the same
   split: actors are graph nodes, every video contributes edges between its
   actors, and connected components are assigned atomically to one split.

DFD naming convention used to derive actors:
  REAL  = NN__scene.mp4                  -> actors {NN}
  FAKE  = NN_MM__scene__HASH.mp4         -> actors {NN (source face), MM (target scene)}

Output:
  --manifest-out  JSON manifest: one record per video with actors, source
                  group (connected component), label, assigned split, origin.
  --summary-out   human-readable split statistics.

The locked test split is created here and must not be used for threshold
selection or hyperparameter tuning (see TRAINING_NOTES.md).
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

REAL_RE = re.compile(r"^(\d+)__(.+)$")
FAKE_RE = re.compile(r"^(\d+)_(\d+)__(.+?)__\w{8}$")

SPLITS = ["train", "val", "calibration", "test"]


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def collect_videos(data_root: Path):
    """Yield dicts describing every video under data_root/<split>/<REAL|FAKE>."""
    records = []
    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = "REAL" if label_dir.name.upper() == "REAL" else "FAKE"
            for video in sorted(label_dir.glob("*.mp4")):
                stem = video.stem
                if label == "REAL":
                    m = REAL_RE.match(stem)
                    if not m:
                        continue
                    actors, scene = [m.group(1)], m.group(2)
                else:
                    m = FAKE_RE.match(stem)
                    if not m:
                        continue
                    actors, scene = [m.group(1), m.group(2)], m.group(3)
                records.append({
                    "video": stem,
                    "path": str(video),
                    "current_split": split_dir.name,
                    "label": label,
                    "actors": actors,
                    "scene": scene,
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-controlled dataset split")
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ and val/ raw videos")
    parser.add_argument("--mode", choices=["scene", "actor"], default="scene",
                        help="scene: each original scene + all its manipulated derivatives stay "
                             "together (fine-grained, prevents related-source leakage; face "
                             "identities may still cross splits). actor: connected components "
                             "over the actor graph (strictest identity-disjointness, but with "
                             "dense DFD pair coverage this collapses into very few, huge groups).")
    parser.add_argument("--manifest-out", default="dataset_manifest.json")
    parser.add_argument("--summary-out", default="SPLIT_SUMMARY.md")
    parser.add_argument("--fractions", default="0.70,0.15,0.05,0.10",
                        help="Comma list for train,val,calibration,test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fractions = [float(x) for x in args.fractions.split(",")]
    assert len(fractions) == 4 and abs(sum(fractions) - 1.0) < 1e-6, "fractions must be 4 values summing to 1"

    records = collect_videos(Path(args.data_dir))
    if not records:
        raise SystemExit("No videos found; check --data-dir")

    # ---- connected components over grouping keys ------------------------------
    uf = UnionFind()
    for r in records:
        if args.mode == "scene":
            # Group purely by scene name: the original video and every manipulated
            # derivative sharing that scene land in one group regardless of the
            # dataset's source/target naming convention.
            uf.union("scene::" + r["scene"], "scene::" + r["scene"])
        else:
            first = "actor::" + r["actors"][0]
            for a in r["actors"]:
                uf.union(first, "actor::" + a)
    for r in records:
        key = ("actor::" + r["actors"][0]) if args.mode == "actor" else ("scene::" + r["scene"])
        r["source_group"] = f"{args.mode}-group-{uf.find(key)}"

    groups = defaultdict(list)
    for r in records:
        groups[r["source_group"]].append(r)

    # ---- assign whole groups to splits, weighted by video count --------------
    rng = random.Random(args.seed)
    group_items = sorted(groups.items(), key=lambda kv: kv[0])
    total = len(records)
    targets = {s: f * total for s, f in zip(SPLITS, fractions)}
    assigned = {s: 0 for s in SPLITS}
    # largest groups first -> better chance to hit every split
    ordered = sorted(group_items, key=lambda kv: len(kv[1]), reverse=True)
    for gid, vids in ordered:
        deficits = {s: targets[s] - assigned[s] for s in SPLITS}
        # choose the split with the largest remaining deficit, tie-broken randomly
        best = max(SPLITS, key=lambda s: (deficits[s], rng.random()))
        for r in vids:
            r["assigned_split"] = best
        assigned[best] += len(vids)

    # ---- summary --------------------------------------------------------------
    by_split_label = defaultdict(lambda: defaultdict(int))
    for r in records:
        by_split_label[r["assigned_split"]][r["label"]] += 1

    lines = [
        "# Leakage-Controlled Split Summary",
        "",
        f"- Source data root: `{args.data_dir}`",
        f"- Grouping mode: **{args.mode}**",
        f"- Videos: {len(records)} (REAL={sum(1 for r in records if r['label'] == 'REAL')}, "
        f"FAKE={sum(1 for r in records if r['label'] == 'FAKE')})",
        f"- Source groups: {len(groups)}",
        f"- Seed: {args.seed} (deterministic for the same input)",
        "",
        "| Split | REAL | FAKE | Total | Share |",
        "|-------|------|------|-------|-------|",
    ]
    for s in SPLITS:
        rl = by_split_label[s]["REAL"]
        fk = by_split_label[s]["FAKE"]
        lines.append(f"| {s} | {rl} | {fk} | {rl + fk} | {(rl + fk) / len(records):.1%} |")

    if args.mode == "scene":
        lines += [
            "",
            "## Leakage properties (scene mode)",
            "",
            "- Prevented: original/manipulated pairs across splits — every video sharing a",
            "  scene name (the original plus ALL manipulated derivatives of it) is assigned",
            "  to one split atomically, regardless of which actor convention the dataset uses.",
            "- Remaining limitation: face identities are NOT disjoint across splits (the same",
            "  actor appears in multiple scenes); run `--mode actor` for strict",
            "  identity-disjointness and accept far fewer, larger groups.",
        ]
    else:
        lines += [
            "",
            "## Leakage properties (actor mode)",
            "",
            "- Prevented: face-identity and scene-identity leakage (connected components over",
            "  the actor graph; a manipulated video and both of its identities stay together).",
            "- Remaining limitation: dense source/target pair coverage in DFD collapses the",
            "  graph into very few components, so splits are coarse and per-split diversity",
            "  is limited.",
        ]
    lines += [
        "",
        f"- The test split is LOCKED: use it only for final reporting, never for threshold",
        f"  selection or hyperparameter tuning (calibration split is for threshold calibration).",
    ]

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "mode": args.mode,
        "fractions": dict(zip(SPLITS, fractions)),
        "split_method": (
            "scene-connected components (original scene + all manipulated derivatives in one split)"
            if args.mode == "scene"
            else "connected components over actor graph (face + scene identities)"
        ),
        "videos": [
            {k: r[k] for k in ("video", "path", "label", "actors", "source_group",
                               "current_split", "assigned_split")}
            for r in sorted(records, key=lambda r: r["video"])
        ],
    }

    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[split] manifest -> {args.manifest_out}")
    print(f"[split] summary  -> {args.summary_out}")


if __name__ == "__main__":
    main()
