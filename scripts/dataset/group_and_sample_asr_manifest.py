import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# =========================
# CONFIG
# =========================
MANIFEST_IN  = Path("data/processed/manifests/asr_manifest.jsonl")
MANIFEST_OUT = Path("data/processed/manifests/asr_manifest_grouped.jsonl")
SAMPLE_OUT   = Path("data/processed/manifests/asr_manifest_grouped_sample.jsonl")

MIN_DURATION  = 5.0   # merge utterances shorter than this (seconds)
MAX_DURATION  = 30.0  # never exceed this (Whisper's sweet spot is 10-30s)
GAP_THRESHOLD = 3.0   # don't merge if gap between utterances exceeds this (seconds)

# If True, drop grouped records still shorter than MIN_DURATION after grouping
# (isolated utterances with no same-speaker neighbour within GAP_THRESHOLD).
DROP_BELOW_MIN_DURATION = False

SAMPLE_N    = 20000   # number of records in the sample
SAMPLE_SEED = 42


# =========================
# HELPERS
# =========================
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    first, last = group[0], group[-1]
    # Suffix merged groups so the ID is distinguishable from an ungrouped utterance
    # and can't silently collide when joined against the original manifest.
    seg_id = first["segment_id"] if len(group) == 1 else f"{first['segment_id']}__grp{len(group)}"
    return {
        "meeting_id":       first["meeting_id"],
        "segment_id":       seg_id,
        "speaker_id":       first["speaker_id"],
        "start_time":       first["start_time"],
        "end_time":         last["end_time"],
        "duration":         last["end_time"] - first["start_time"],
        "audio":            first["audio"],
        "reference_text":   " ".join(r.get("reference_text", "") for r in group).strip(),
        "overlap":          any(r.get("overlap", False) for r in group),
        "grouped_segments": [r["segment_id"] for r in group],
        "n_merged":         len(group),
    }


# =========================
# GROUPING LOGIC
# =========================
def group_utterances(records, min_dur, max_dur, gap_thresh):
    records = sorted(records, key=lambda r: (r["meeting_id"], r["speaker_id"], r["start_time"]))
    grouped = []
    current_group = []

    for rec in records:
        dur = float(rec.get("duration") or rec["end_time"] - rec["start_time"])

        if not current_group:
            current_group.append(rec)
            continue

        prev  = current_group[-1]
        start = float(current_group[0]["start_time"])
        same_meeting = rec["meeting_id"] == prev["meeting_id"]
        same_speaker  = rec["speaker_id"] == prev["speaker_id"]
        gap           = float(rec["start_time"]) - float(prev["end_time"])
        # Prospective span includes the gap we'd be absorbing, not just the utterance durations.
        would_exceed  = (float(rec["end_time"]) - start) > max_dur

        # Gaps over gap_thresh always flush the group: the audio in the gap belongs
        # to other speakers on the Mix-Headset channel, and merging across it pulls
        # foreign speech into the group (inflates ASR insertions vs. reference text).
        can_merge = (
            same_meeting
            and same_speaker
            and not would_exceed
            and gap <= gap_thresh
        )

        if can_merge:
            current_group.append(rec)
        else:
            grouped.append(merge_group(current_group))
            current_group = [rec]

    if current_group:
        grouped.append(merge_group(current_group))

    return grouped


# =========================
# VALIDATION
# =========================
def validate_grouped(grouped, original_records):
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)

    original_index = {r["segment_id"]: r for r in original_records}
    issues = []

    # 1. Duration bounds
    over_max  = [r for r in grouped if r["duration"] > MAX_DURATION]
    zero_dur  = [r for r in grouped if r["duration"] <= 0]
    if over_max:
        issues.append(f"[FAIL] {len(over_max)} groups exceed MAX_DURATION ({MAX_DURATION}s)")
    else:
        print(f"[OK]   No groups exceed MAX_DURATION ({MAX_DURATION}s)")
    if zero_dur:
        issues.append(f"[FAIL] {len(zero_dur)} groups have zero/negative duration")
    else:
        print(f"[OK]   All groups have positive duration")

    # 2. Audio path consistency within groups
    multi_audio = []
    for g in grouped:
        if g["n_merged"] > 1:
            paths = set()
            for seg_id in g["grouped_segments"]:
                orig = original_index.get(seg_id)
                if orig:
                    paths.add(orig["audio"]["path"])
            if len(paths) > 1:
                multi_audio.append(g["segment_id"])
    if multi_audio:
        issues.append(f"[FAIL] {len(multi_audio)} groups span multiple audio files — will fail during ASR")
        for s in multi_audio[:5]:
            print(f"       Example: {s}")
    else:
        print(f"[OK]   All groups use a single audio file")

    # 3. Missing audio path
    missing_audio = [r for r in grouped if not r.get("audio", {}).get("path")]
    if missing_audio:
        issues.append(f"[FAIL] {len(missing_audio)} groups have missing audio path")
    else:
        print(f"[OK]   All groups have a valid audio path")

    # 4. Empty reference text
    empty_text = [r for r in grouped if not r.get("reference_text", "").strip()]
    print(f"[INFO] {len(empty_text)} groups have empty reference_text "
          f"({len(empty_text)/len(grouped)*100:.1f}%) — may be silence/noise segments")

    # 5. Segment ID coverage
    original_seg_ids  = set(r["segment_id"] for r in original_records)
    grouped_seg_ids   = set(seg for g in grouped for seg in g["grouped_segments"])
    missing_in_groups = original_seg_ids - grouped_seg_ids
    extra_in_groups   = grouped_seg_ids - original_seg_ids
    if missing_in_groups:
        issues.append(f"[FAIL] {len(missing_in_groups)} original segments missing from grouped output")
    else:
        print(f"[OK]   All original segments accounted for in grouped output")
    if extra_in_groups:
        issues.append(f"[FAIL] {len(extra_in_groups)} segment IDs in groups not found in original")
    else:
        print(f"[OK]   No unknown segment IDs in grouped output")

    # Summary
    print()
    if issues:
        print(f"ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("All checks passed.")
    print("=" * 50)

    return len(issues) == 0


# =========================
# STATS
# =========================
def print_stats(label, records):
    durations = [r["duration"] for r in records]
    print(f"\n{label} ({len(records)} records):")
    print(f"  Mean   : {np.mean(durations):.2f}s")
    print(f"  Median : {np.median(durations):.2f}s")
    print(f"  p90    : {np.percentile(durations, 90):.2f}s")
    print(f"  Min    : {min(durations):.2f}s")
    print(f"  Max    : {max(durations):.2f}s")

    buckets = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    prev = 0.0
    for b in buckets:
        count = sum(1 for d in durations if prev <= d < b)
        pct = count / len(durations) * 100
        print(f"  {prev:.1f}s – {b:.1f}s : {count:>6} ({pct:.1f}%)")
    count = sum(1 for d in durations if d >= 25.0)
    print(f"  25.0s+      : {count:>6} ({count/len(durations)*100:.1f}%)")

    if "n_merged" in records[0]:
        print(f"  Single-segment : {sum(1 for r in records if r['n_merged'] == 1)}")
        print(f"  Multi-segment  : {sum(1 for r in records if r['n_merged'] > 1)}")
        print(f"  Max merged     : {max(r['n_merged'] for r in records)}")


# =========================
# SAMPLING
# =========================
def duration_bucket(d):
    # Buckets tuned for post-grouping distribution (most rows are >= MIN_DURATION).
    if d < 10:
        return "short"    # 5-10s
    elif d < 15:
        return "medium"   # 10-15s
    elif d < 20:
        return "long"     # 15-20s
    else:
        return "xlong"    # 20-25s

def sample_grouped(records, n, seed):
    random.seed(seed)
    total = len(records)

    if n >= total:
        print(f"Requested {n} >= total {total}, using full manifest.")
        return records

    # Stratify by (duration bucket, overlap)
    strata = defaultdict(list)
    for i, r in enumerate(records):
        key = (duration_bucket(r["duration"]), bool(r.get("overlap", False)))
        strata[key].append(i)

    sampled_indices = set()
    stratum_counts = {}
    for key, indices in strata.items():
        proportion = len(indices) / total
        k = min(round(proportion * n), len(indices))
        chosen = random.sample(indices, k)
        sampled_indices.update(chosen)
        stratum_counts[key] = k

    # Fill to exactly n
    if len(sampled_indices) < n:
        pool = [i for i in range(total) if i not in sampled_indices]
        sampled_indices.update(random.sample(pool, n - len(sampled_indices)))

    # Trim to exactly n
    if len(sampled_indices) > n:
        sampled_indices = set(random.sample(list(sampled_indices), n))

    sampled = [records[i] for i in sorted(sampled_indices)]
    random.shuffle(sampled)

    print(f"\nSample stratum breakdown:")
    for (bucket, overlap), count in sorted(stratum_counts.items()):
        original = len(strata[(bucket, overlap)])
        print(f"  {bucket:<8} overlap={overlap}  {count:>5} / {original:>6}  ({count/original*100:.1f}%)")

    return sampled


# =========================
# RUN
# =========================
print("Loading manifest...")
records = load_jsonl(MANIFEST_IN)
print_stats("Original", [{"duration": float(r.get("duration") or r["end_time"] - r["start_time"])} for r in records])

print("\nGrouping utterances...")
# Filter out zero/negative duration segments before grouping
n_before = len(records)
records_valid = [r for r in records if float(r.get("duration") or r["end_time"] - r["start_time"]) > 0]
n_filtered = n_before - len(records_valid)
if n_filtered:
    print(f"Filtered {n_filtered} zero/negative duration segments before grouping")

grouped = group_utterances(records_valid, MIN_DURATION, MAX_DURATION, GAP_THRESHOLD)

# Correct only sub-millisecond floating-point drift; crash on meaningful violations
# so a grouping bug can't silently shrink real audio spans.
FP_TOLERANCE = 1e-3
real_over = [r for r in grouped if r["duration"] > MAX_DURATION + FP_TOLERANCE]
if real_over:
    raise RuntimeError(
        f"{len(real_over)} groups exceed MAX_DURATION ({MAX_DURATION}s) by more than "
        f"{FP_TOLERANCE}s — grouping invariant violated. Example: {real_over[0]['segment_id']} "
        f"({real_over[0]['duration']:.6f}s)"
    )
fp_drift = [r for r in grouped if r["duration"] > MAX_DURATION]
if fp_drift:
    print(f"Correcting {len(fp_drift)} groups with sub-ms FP drift over MAX_DURATION")
    for r in fp_drift:
        r["end_time"] = r["start_time"] + MAX_DURATION
        r["duration"] = MAX_DURATION
print_stats("Grouped", grouped)

if DROP_BELOW_MIN_DURATION:
    n_before_drop = len(grouped)
    kept_seg_ids = {s for g in grouped if g["duration"] >= MIN_DURATION for s in g["grouped_segments"]}
    grouped = [g for g in grouped if g["duration"] >= MIN_DURATION]
    n_dropped = n_before_drop - len(grouped)
    print(f"\nDropped {n_dropped} groups below MIN_DURATION ({MIN_DURATION}s): "
          f"{n_dropped / n_before_drop * 100:.1f}% of groups")
    print_stats("Grouped (after drop)", grouped)
    # Align validation set so the segment-coverage check reflects what should now be present.
    validation_records = [r for r in records_valid if r["segment_id"] in kept_seg_ids]
else:
    validation_records = records_valid

valid = validate_grouped(grouped, validation_records)

if valid:
    write_jsonl(MANIFEST_OUT, grouped)
    print(f"\nGrouped manifest written to {MANIFEST_OUT}")

    print(f"\nCreating sample of {SAMPLE_N} records...")
    sampled = sample_grouped(grouped, SAMPLE_N, SAMPLE_SEED)
    write_jsonl(SAMPLE_OUT, sampled)
    print(f"Sample written to {SAMPLE_OUT} ({len(sampled)} records)")
else:
    print("\nGrouped manifest NOT written due to validation failures — fix issues first.")