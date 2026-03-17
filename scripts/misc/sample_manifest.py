#!/usr/bin/env python3
"""
Create a smaller (optionally stratified) sample from a JSONL NER manifest.

Input JSONL lines should look like (minimum):
{
  "meeting_id": "...",
  "segment_id": "...",
  "text": "...",
  "entities": [...],   # optional; if missing, treated as empty
  "overlap": false     # optional; if missing, treated as false
}

Output: JSONL with the sampled rows (same schema as input).

Sampling modes:
- random (default)
- stratified by:
    * has_entities  (0/1)
    * overlap       (0/1)
    * length_bucket (short/medium/long by char length)

Examples:
  # random 2000
  python sample_manifest.py --manifest_in data/processed/manifests/ner_manifest.jsonl \
    --manifest_out data/processed/manifests/ner_manifest_sample2k.jsonl \
    --n 2000 --seed 13

  # stratified 2000 (recommended for LLM runs)
  python sample_manifest.py --manifest_in data/processed/manifests/ner_gold_reference.jsonl \
    --manifest_out data/processed/manifests/ner_gold_reference_sample2k.jsonl \
    --n 2000 --seed 13 --stratify has_entities overlap length_bucket

  # sample only utterances that have at least 1 gold entity
  python sample_manifest.py --manifest_in ... --manifest_out ... \
    --n 2000 --only_with_entities
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def has_entities(rec: Dict[str, Any]) -> int:
    ents = rec.get("entities", [])
    return 1 if isinstance(ents, list) and len(ents) > 0 else 0


def get_overlap(rec: Dict[str, Any]) -> int:
    return 1 if bool(rec.get("overlap", False)) else 0


def length_bucket(text: str, short_max: int, medium_max: int) -> str:
    n = len(text)
    if n <= short_max:
        return "short"
    if n <= medium_max:
        return "medium"
    return "long"


def strat_key(
    rec: Dict[str, Any],
    stratify: List[str],
    short_max: int,
    medium_max: int,
) -> Tuple[Any, ...]:
    text = (rec.get("text", "") or "")
    key_parts: List[Any] = []

    for s in stratify:
        if s == "has_entities":
            key_parts.append(has_entities(rec))
        elif s == "overlap":
            key_parts.append(get_overlap(rec))
        elif s == "length_bucket":
            key_parts.append(length_bucket(text, short_max=short_max, medium_max=medium_max))
        else:
            raise ValueError(f"Unknown stratify field: {s}")

    return tuple(key_parts)


def proportional_alloc(counts: Dict[Tuple[Any, ...], int], n_total: int) -> Dict[Tuple[Any, ...], int]:
    """
    Allocate sample sizes to strata proportionally, ensuring sum == n_total.
    """
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}

    # initial floor allocations
    alloc = {}
    remainders = []
    for k, c in counts.items():
        exact = (c / total) * n_total
        a = int(math.floor(exact))
        alloc[k] = a
        remainders.append((exact - a, k))

    # distribute remaining by largest remainder
    remaining = n_total - sum(alloc.values())
    remainders.sort(reverse=True, key=lambda x: x[0])
    for _, k in remainders[:remaining]:
        alloc[k] += 1

    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_in", required=True, type=Path)
    ap.add_argument("--manifest_out", required=True, type=Path)
    ap.add_argument("--n", type=int, required=True, help="Number of rows to sample")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument(
        "--stratify",
        nargs="*",
        default=[],
        choices=["has_entities", "overlap", "length_bucket"],
        help="Optional stratification fields (recommended for LLM benchmarking).",
    )

    ap.add_argument("--short_max", type=int, default=80, help="Max chars for 'short' bucket (default 80)")
    ap.add_argument("--medium_max", type=int, default=200, help="Max chars for 'medium' bucket (default 200)")

    ap.add_argument(
        "--only_with_entities",
        action="store_true",
        help="If set, only sample from rows where entities list is non-empty.",
    )

    ap.add_argument(
        "--dedupe_by_segment_id",
        action="store_true",
        help="If set, keep only the first occurrence of a segment_id before sampling.",
    )

    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load
    rows: List[Dict[str, Any]] = list(iter_jsonl(args.manifest_in))

    # Optional dedupe
    if args.dedupe_by_segment_id:
        seen = set()
        deduped = []
        for r in rows:
            sid = r.get("segment_id")
            if sid is None or sid not in seen:
                deduped.append(r)
                if sid is not None:
                    seen.add(sid)
        rows = deduped

    # Optional filter: only rows with entities
    if args.only_with_entities:
        rows = [r for r in rows if has_entities(r) == 1]

    if not rows:
        raise SystemExit("No rows available after filtering/deduping.")

    if args.n > len(rows):
        raise SystemExit(f"Requested n={args.n}, but only {len(rows)} rows available.")

    # Random sampling (no stratification)
    if not args.stratify:
        sampled = rng.sample(rows, args.n)
        write_jsonl(args.manifest_out, sampled)
        print(f"Wrote random sample: {len(sampled)} rows -> {args.manifest_out}")
        return

    # Stratified sampling
    strata: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        k = strat_key(r, args.stratify, short_max=args.short_max, medium_max=args.medium_max)
        strata[k].append(r)

    counts = {k: len(v) for k, v in strata.items()}
    alloc = proportional_alloc(counts, args.n)

    # Sample within each stratum up to available
    sampled: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for k, group in strata.items():
        rng.shuffle(group)
        take = min(alloc.get(k, 0), len(group))
        sampled.extend(group[:take])
        leftovers.extend(group[take:])

    # If some strata were too small, top up from leftovers randomly
    if len(sampled) < args.n:
        need = args.n - len(sampled)
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:need])

    # If we somehow overshot, trim (shouldn’t happen, but safe)
    if len(sampled) > args.n:
        rng.shuffle(sampled)
        sampled = sampled[: args.n]

    # Stable-ish output order (optional): sort by meeting_id, segment_id
    sampled.sort(key=lambda r: (r.get("meeting_id", ""), r.get("segment_id", "")))

    write_jsonl(args.manifest_out, sampled)
    print(
        f"Wrote stratified sample: {len(sampled)} rows -> {args.manifest_out}\n"
        f"Stratify fields: {args.stratify}\n"
        f"Buckets: {len(strata)}"
    )


if __name__ == "__main__":
    main()