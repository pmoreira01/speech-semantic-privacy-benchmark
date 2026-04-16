import argparse
import random
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import iter_jsonl, write_jsonl


def duration_bucket(d):
    if d < 1:
        return "short"
    elif d < 5:
        return "medium"
    else:
        return "long"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_in", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    records = list(iter_jsonl(Path(args.manifest_in)))

    # Stratify by (duration bucket, overlap)
    strata = {}

    for r in records:
        d = float(r.get("duration", 0))
        bucket = duration_bucket(d)
        overlap = bool(r.get("overlap", False))

        key = (bucket, overlap)
        strata.setdefault(key, []).append(r)

    # Calculate how many per stratum
    total = len(records)
    sampled = []

    for key, group in strata.items():
        proportion = len(group) / total
        k = int(proportion * args.n)

        sampled.extend(random.sample(group, min(k, len(group))))

    # If we are short due to rounding, fill randomly
    if len(sampled) < args.n:
        remaining = args.n - len(sampled)
        remaining_pool = [r for r in records if r not in sampled]
        sampled.extend(random.sample(remaining_pool, remaining))

    # Shuffle final sample
    random.shuffle(sampled)

    write_jsonl(Path(args.manifest_out), sampled)

    print(f"Sampled {len(sampled)} / {len(records)} → {args.manifest_out}")


if __name__ == "__main__":
    main()