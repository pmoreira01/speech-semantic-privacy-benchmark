#!/usr/bin/env python3
"""
Generate an NER manifest JSONL from canonical truth meeting files.

Input:
  data/processed/meetings/*.truth.json

Output:
  data/processed/manifests/ner_gold_reference.jsonl (default)

Each line is one text unit (utterance):
  {
    "meeting_id": "...",
    "segment_id": "...",           # utterance_id
    "speaker_id": "...",
    "start_time": float,
    "end_time": float,
    "text": "...",
    "entities": [
      {"start_char": int, "end_char": int, "label": "...", "text": "...", "entity_id": "..."}
    ],
    "overlap": bool
  }

Options:
- --label-map: JSON mapping label_id -> label (e.g., "ne_1113" -> "MARKETING")
- --label-policy: keep_id | map | drop_unmapped
- --min-entities: require at least N entities in utterance (default 0)

This script validates entity offsets by default and drops invalid ones (or can fail-fast).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def iter_truth_files(truth_dir: Path) -> Iterable[Path]:
    for p in sorted(truth_dir.glob("*.truth.json")):
        if p.is_file():
            yield p


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_label_map(label_map_file: Optional[Path]) -> Dict[str, str]:
    """
    label_map.json example:
      {
        "ne_1113": "MARKETING",
        "ne_0001": "PERSON",
        ...
      }
    """
    if not label_map_file:
        return {}
    if not label_map_file.exists():
        raise FileNotFoundError(f"--label-map not found: {label_map_file}")
    return json.loads(label_map_file.read_text(encoding="utf-8"))


def validate_entity(text: str, ent: dict) -> Tuple[bool, str]:
    try:
        s = int(ent["start_char"])
        e = int(ent["end_char"])
    except Exception:
        return False, "non-integer offsets"

    if s < 0 or e < 0 or e < s or e > len(text):
        return False, "offset out of bounds"
    sub = text[s:e]
    gold = (ent.get("text") or "")
    if gold and sub != gold:
        # Not always fatal (could be normalization differences), but usually indicates drift.
        return False, f"text mismatch '{sub}' != '{gold}'"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", type=Path, default=Path("data/processed/meetings"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/manifests/ner_manifest.jsonl"))

    ap.add_argument(
        "--label-policy",
        choices=["keep_id", "map", "drop_unmapped"],
        default="keep_id",
        help=(
            "keep_id: keep original label id (e.g., ne_1113). "
            "map: replace using --label-map if available (fallback to id). "
            "drop_unmapped: drop entities whose label_id not in --label-map."
        ),
    )
    ap.add_argument("--label-map", type=Path, default=None, help="JSON mapping label_id -> readable label.")
    ap.add_argument("--min-entities", type=int, default=0, help="Skip utterances with fewer than N entities.")
    ap.add_argument("--skip-overlap", action="store_true", help="Exclude utterances marked overlap=true.")
    ap.add_argument("--fail-fast", action="store_true", help="Fail on invalid entity offsets instead of dropping.")
    ap.add_argument("--meetings", nargs="*", default=None, help="Optional list of meeting_ids to include.")

    args = ap.parse_args()

    truth_dir: Path = args.truth_dir
    if not truth_dir.exists():
        raise FileNotFoundError(f"truth-dir does not exist: {truth_dir}")

    label_map = load_label_map(args.label_map)

    rows: List[dict] = []
    for tf in iter_truth_files(truth_dir):
        meeting = load_json(tf)
        meeting_id = meeting["meeting_id"]

        if args.meetings and meeting_id not in set(args.meetings):
            continue

        meeting_rows: List[dict] = []
        meeting_has_any_entity = False

        for utt in meeting.get("utterances", []):
            if args.skip_overlap and bool(utt.get("overlap", False)):
                continue

            text = (utt.get("text") or "")
            if not text.strip():
                continue

            ents_out: List[dict] = []
            for ent in utt.get("entities", []):
                # label extraction
                label_id = ent.get("label") or ent.get("label_id") or "UNKNOWN"

                if args.label_policy == "keep_id":
                    label = label_id
                elif args.label_policy == "map":
                    label = label_map.get(label_id, label_id)
                else:  # drop_unmapped
                    if label_id not in label_map:
                        continue
                    label = label_map[label_id]

                ok, reason = validate_entity(text, ent)
                if not ok:
                    if args.fail_fast:
                        raise RuntimeError(
                            f"Invalid entity in {meeting_id} {utt.get('utterance_id')}: {reason}. Entity={ent}"
                        )
                    continue

                ents_out.append(
                    {
                        "entity_id": ent.get("entity_id"),
                        "start_char": int(ent["start_char"]),
                        "end_char": int(ent["end_char"]),
                        "label": label,
                        "text": text[int(ent["start_char"]) : int(ent["end_char"])],
                    }
                )

            # Track whether this meeting has any *valid* entity anywhere
            if ents_out:
                meeting_has_any_entity = True

            if len(ents_out) < args.min_entities:
                continue

            meeting_rows.append(
                {
                    "meeting_id": meeting_id,
                    "segment_id": utt["utterance_id"],
                    "speaker_id": utt.get("speaker_id"),
                    "start_time": float(utt["start_time"]),
                    "end_time": float(utt["end_time"]),
                    "text": text,
                    "entities": ents_out,
                    "overlap": bool(utt.get("overlap", False)),
                }
            )

        # Only include this meeting if it has at least one annotated entity
        if not meeting_has_any_entity:
            continue

        rows.extend(meeting_rows)

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} NER examples -> {args.out}")


if __name__ == "__main__":
    main()