#!/usr/bin/env python3
"""
Generate an ASR manifest JSONL from canonical truth meeting files.

Input:
  data/processed/meetings/*.truth.json

Output:
  data/processed/manifests/asr_manifest.jsonl  (default)

Each line is one segment:
  {
    "meeting_id": "...",
    "segment_id": "...",        # utterance_id
    "speaker_id": "...",
    "start_time": float,
    "end_time": float,
    "duration": float,
    "audio": {
      "condition": "IHM",
      "path": "..."
    },
    "reference_text": "...",
    "overlap": bool
  }

Notes:
- You must provide --audio-path-template or --audio-map-file so the manifest knows where audio lives.
- This script does NOT modify text. Keep normalization as a separate downstream step to avoid offset issues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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


def load_audio_map(audio_map_file: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    audio_map.json format:
      {
        "ES2002a": {"IHM": ".../ES2002a.IHM.wav", "SDM": ".../ES2002a.SDM.wav"},
        "ES2002b": {"IHM": "..."}
      }
    """
    if not audio_map_file:
        return {}
    if not audio_map_file.exists():
        raise FileNotFoundError(f"--audio-map-file not found: {audio_map_file}")
    return json.loads(audio_map_file.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth-dir", type=Path, default=Path("data/processed/meetings"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/manifests/asr_manifest.jsonl"))

    # Audio resolution options:
    ap.add_argument(
        "--audio-condition",
        type=str,
        default="IHM",
        help="Audio condition label to write into manifest (e.g., IHM, SDM, MDM).",
    )
    ap.add_argument(
        "--audio-path-template",
        type=str,
        default=None,
        help=(
            "Template to build audio path from meeting_id, e.g. "
            "'data/raw/ami/{meeting_id}/audio/{condition}.wav' or "
            "'data/raw/ami/audio/{meeting_id}.{condition}.wav'. "
            "Variables: {meeting_id}, {condition}"
        ),
    )
    ap.add_argument(
        "--audio-map-file",
        type=Path,
        default=None,
        help="JSON file mapping meeting_id -> {condition: path}. Overrides --audio-path-template if present.",
    )

    # Filtering
    ap.add_argument("--min-duration", type=float, default=0.0, help="Skip utterances shorter than this (seconds).")
    ap.add_argument("--max-duration", type=float, default=60.0, help="Skip utterances longer than this (seconds).")
    ap.add_argument("--skip-overlap", action="store_true", help="If set, exclude utterances marked overlap=true.")
    ap.add_argument("--meetings", nargs="*", default=None, help="Optional list of meeting_ids to include.")

    args = ap.parse_args()

    truth_dir: Path = args.truth_dir
    if not truth_dir.exists():
        raise FileNotFoundError(f"truth-dir does not exist: {truth_dir}")

    audio_map = load_audio_map(args.audio_map_file)

    rows: List[dict] = []
    for tf in iter_truth_files(truth_dir):
        meeting = load_json(tf)
        meeting_id = meeting["meeting_id"]

        if args.meetings and meeting_id not in set(args.meetings):
            continue

        # Resolve audio path:
        audio_path = None
        if audio_map:
            audio_path = audio_map.get(meeting_id, {}).get(args.audio_condition)
        if audio_path is None and args.audio_path_template:
            audio_path = args.audio_path_template.format(meeting_id=meeting_id, condition=args.audio_condition)

        if audio_path is None:
            raise RuntimeError(
                f"No audio path could be resolved for meeting {meeting_id}. "
                f"Provide --audio-path-template or --audio-map-file."
            )

        for utt in meeting.get("utterances", []):
            start = float(utt["start_time"])
            end = float(utt["end_time"])
            dur = max(0.0, end - start)

            if dur < args.min_duration or dur > args.max_duration:
                continue
            if args.skip_overlap and bool(utt.get("overlap", False)):
                continue

            ref_text = (utt.get("text") or "").strip()
            if not ref_text:
                continue

            rows.append(
                {
                    "meeting_id": meeting_id,
                    "segment_id": utt["utterance_id"],
                    "speaker_id": utt.get("speaker_id"),
                    "start_time": start,
                    "end_time": end,
                    "duration": dur,
                    "audio": {
                        "condition": args.audio_condition,
                        "path": audio_path,
                    },
                    "reference_text": ref_text,
                    "overlap": bool(utt.get("overlap", False)),
                }
            )

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} ASR segments -> {args.out}")


if __name__ == "__main__":
    main()