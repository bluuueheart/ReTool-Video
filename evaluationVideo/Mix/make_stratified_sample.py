#!/usr/bin/env python3
"""Draw a seeded, subset-proportional sample from each enabled dataset and preflight the videos.

Per dataset, ``--per-dataset`` samples are allocated across its subsets proportionally to the subset sizes
(largest-remainder rounding). Inside a subset, samples are drawn round-robin over ``duration`` (Video-MME) so the
short/medium/long mix stays balanced; other datasets draw uniformly.

Outputs (under ``--out-dir``): ``uids.txt`` (for ``run_mix_batch.py --sample-uids @uids.txt``), ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Retool_Video_RL"))

from run_mix_batch import load_config, load_inventories  # noqa: E402


def allocate(sizes: Dict[str, int], total: int) -> Dict[str, int]:
    population = sum(sizes.values())
    exact = {name: total * size / population for name, size in sizes.items()}
    alloc = {name: min(sizes[name], int(value)) for name, value in exact.items()}
    remainder = total - sum(alloc.values())
    for name in sorted(sizes, key=lambda n: (exact[n] - int(exact[n]), sizes[n]), reverse=True):
        if remainder <= 0:
            break
        if alloc[name] < sizes[name]:
            alloc[name] += 1
            remainder -= 1
    return alloc


def draw(samples: List, count: int, rng: random.Random, duration_counts: Dict[str, int]) -> List:
    """Pick ``count`` samples; each pick takes the duration bucket that is globally least represented so far."""
    by_duration: Dict[str, List] = defaultdict(list)
    for sample in samples:
        by_duration[str((sample.official_meta or {}).get("duration") or "")].append(sample)
    for group in by_duration.values():
        rng.shuffle(group)
    picked: List = []
    while len(picked) < count and any(by_duration.values()):
        keys = [key for key, group in by_duration.items() if group]
        rng.shuffle(keys)
        key = min(keys, key=lambda k: duration_counts[k])
        picked.append(by_duration[key].pop())
        duration_counts[key] += 1
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--counts", default="", help="per-dataset counts, e.g. mvbench=33,mlvu=33,videomme=34 (overrides --per-dataset)")
    parser.add_argument("--name", default="uids", help="output stem: <name>.txt and <name>_summary.json")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    counts = {k.strip(): int(v) for k, v in (item.split("=") for item in args.counts.split(",") if item.strip())}
    config = load_config(Path(args.config))
    inventories = load_inventories(config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uids: List[str] = []
    summary: Dict[str, dict] = {}
    missing: List[str] = []
    for inventory in inventories:
        rng = random.Random(f"{args.seed}:{inventory.dataset_name}")
        # Only samples whose video file is on disk can be drawn (e.g. MVBench Fine-grained Pose needs NTU RGB+D
        # videos that are not part of the public zips); unavailable counts are reported in the summary.
        available = {
            name: [sample for sample in inventory.samples_by_subset.get(name, ()) if Path(sample.video_path).exists()]
            for name in inventory.ordered_subset_names
        }
        unavailable = {
            name: len(inventory.samples_by_subset.get(name, ())) - len(items)
            for name, items in available.items()
            if len(inventory.samples_by_subset.get(name, ())) - len(items)
        }
        sizes = {name: len(items) for name, items in available.items() if items}
        alloc = allocate(sizes, min(counts.get(inventory.dataset_name, args.per_dataset), sum(sizes.values())))
        chosen = []
        duration_counts: Dict[str, int] = defaultdict(int)
        for name, count in alloc.items():
            chosen.extend(draw(list(available[name]), count, rng, duration_counts))
        durations: Dict[str, int] = defaultdict(int)
        for sample in chosen:
            durations[str((sample.official_meta or {}).get("duration") or "-")] += 1
            if not Path(sample.video_path).exists():
                missing.append(sample.sample_uid)
        uids.extend(sample.sample_uid for sample in chosen)
        summary[inventory.dataset_name] = {
            "population": sum(sizes.values()),
            "sampled": len(chosen),
            "subset_population": sizes,
            "unavailable_videos_by_subset": unavailable,
            "subset_sampled": {name: count for name, count in alloc.items() if count},
            "duration": dict(durations),
        }

    (out_dir / f"{args.name}.txt").write_text("\n".join(uids) + "\n", encoding="utf-8")
    summary["missing_videos"] = missing
    (out_dir / f"{args.name}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, info in summary.items():
        if name == "missing_videos":
            continue
        print(name, "population", info["population"], "sampled", info["sampled"], "duration", info["duration"])
        print("   ", info["subset_sampled"])
    print("missing videos:", len(missing))


if __name__ == "__main__":
    main()
