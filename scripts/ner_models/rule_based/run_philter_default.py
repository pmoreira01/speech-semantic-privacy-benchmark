#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def recover_phi_spans(original: str, redacted: str) -> List[Dict[str, Any]]:
    """
    Recover approximate redacted spans by diffing original vs redacted text.
    Any changed/deleted region in the original is treated as PHI.
    """
    spans: List[Dict[str, Any]] = []

    matcher = SequenceMatcher(a=original, b=redacted, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            spans.append(
                {
                    "start_char": i1,
                    "end_char": i2,
                    "label": "PHI",
                    "text": original[i1:i2],
                }
            )

    if not spans:
        return spans

    spans.sort(key=lambda x: (x["start_char"], x["end_char"]))
    merged = [spans[0]]

    for s in spans[1:]:
        prev = merged[-1]
        if s["start_char"] <= prev["end_char"]:
            prev["end_char"] = max(prev["end_char"], s["end_char"])
            prev["text"] = original[prev["start_char"]:prev["end_char"]]
        else:
            merged.append(s)

    return merged


def prepare_input_files(manifest_rows: List[Dict[str, Any]], input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    for rec in manifest_rows:
        segment_id = rec["segment_id"]
        text = rec.get("text", "") or ""
        out_file = input_dir / f"{segment_id}.txt"
        out_file.write_text(text, encoding="utf-8")


def build_philter_cmd(
    python_exec: str,
    philter_root: Path,
    input_dir: Path,
    output_dir: Path,
) -> List[str]:
    return [
        python_exec,
        str(philter_root / "main.py"),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-f", str(philter_root / "configs" / "philter_delta.json"),
        "--prod=True",
        "--outputformat", "asterisk",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_in", default="data/processed/manifests/ner_manifest.jsonl")
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--philter_root", required=True, help="Path to cloned philter-ucsf repo")
    ap.add_argument(
        "--work_dir",
        default="data/processed/philter_work",
        help="Temporary working directory for input/output text files",
    )
    ap.add_argument(
        "--python_exec",
        default=sys.executable,
        help="Python executable to use for running Philter (use Philter env if needed)",
    )
    ap.add_argument("--no_text", action="store_true")
    ap.add_argument("--keep_workdir", action="store_true")
    args = ap.parse_args()

    philter_root = Path(args.philter_root).resolve()
    manifest_in = Path(args.manifest_in).resolve()
    manifest_out = Path(args.manifest_out).resolve()
    work_dir = Path(args.work_dir).resolve()

    if not (philter_root / "main.py").exists():
        raise FileNotFoundError(f"Could not find main.py under {philter_root}")

    if not (philter_root / "configs" / "philter_delta.json").exists():
        raise FileNotFoundError(
            f"Could not find default config: {philter_root / 'configs' / 'philter_delta.json'}"
        )

    rows = list(iter_jsonl(manifest_in))
    if not rows:
        raise RuntimeError(f"No rows found in manifest: {manifest_in}")

    input_dir = work_dir / "input"
    output_dir = work_dir / "output"

    if work_dir.exists():
        shutil.rmtree(work_dir)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_input_files(rows, input_dir)

    cmd = build_philter_cmd(
        python_exec=args.python_exec,
        philter_root=philter_root,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    print("Running:", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(philter_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Philter failed")

    avg_latency_ms = total_latency_ms / len(rows)

    out_rows: List[Dict[str, Any]] = []

    for rec in rows:
        segment_id = rec["segment_id"]
        text = rec.get("text", "") or ""

        redacted_file = output_dir / f"{segment_id}.txt"

        out_rec: Dict[str, Any] = {
            "meeting_id": rec.get("meeting_id"),
            "segment_id": rec.get("segment_id"),
            "speaker_id": rec.get("speaker_id"),
            "start_time": rec.get("start_time"),
            "end_time": rec.get("end_time"),
            "overlap": rec.get("overlap"),
            "model": "philter",
            "latency_ms": avg_latency_ms,
            "predicted_entities": [],
        }

        if not args.no_text:
            out_rec["text"] = text

        if not redacted_file.exists():
            out_rec["error"] = f"Missing output file: {redacted_file.name}"
            out_rows.append(out_rec)
            continue

        redacted_text = redacted_file.read_text(encoding="utf-8", errors="replace")
        out_rec["predicted_entities"] = recover_phi_spans(text, redacted_text)
        out_rows.append(out_rec)

    write_jsonl(manifest_out, out_rows)

    print(f"Wrote {len(out_rows)} rows -> {manifest_out}")
    print(f"Total Philter runtime: {total_latency_ms:.2f} ms")
    print(f"Average per utterance: {avg_latency_ms:.2f} ms")

    if not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()