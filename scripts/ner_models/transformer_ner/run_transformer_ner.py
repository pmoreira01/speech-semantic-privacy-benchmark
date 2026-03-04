import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from tqdm import tqdm
from transformers import pipeline


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_bio(label: str) -> str:
    if label.startswith(("B-", "I-")):
        return label[2:]
    return label


def main(
    manifest_in: str,
    manifest_out: str,
    model_name: str = "dslim/bert-base-NER",
    device: int = -1,
    keep_text: bool = True,
):
    """
    Transformer encoder NER (token classification) runner.

    model_name: HF model id (token-classification)
    device: -1 for CPU, 0 for first GPU, etc.
    """

    ner = pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",  # merges subword tokens into spans
        device=device
    )

    out_rows = []
    for rec in tqdm(iter_jsonl(manifest_in), desc=f"NER: {model_name}"):
        text = rec.get("text", "")

        t0 = time.perf_counter()
        preds = ner(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        predicted_entities = []
        for p in preds:
            # pipeline returns fields like: start, end, word, entity_group, score
            start = int(p["start"])
            end = int(p["end"])
            raw_label = p.get("entity_group") or p.get("entity") or "MISC"
            label = strip_bio(str(raw_label))
            score = float(p.get("score", 0.0))

            predicted_entities.append({
                "start_char": start,
                "end_char": end,
                "label": label,  # model label space (PER/ORG/LOC/MISC for many NER models)
                "text": text[start:end],
                "score": score
            })

        out_rec = {
            "meeting_id": rec.get("meeting_id"),
            "segment_id": rec.get("segment_id"),
            "speaker_id": rec.get("speaker_id"),
            "start_time": rec.get("start_time"),
            "end_time": rec.get("end_time"),
            "overlap": rec.get("overlap"),
            "model": model_name,
            "latency_ms": latency_ms,
            "predicted_entities": predicted_entities,
        }
        if keep_text:
            out_rec["text"] = text

        out_rows.append(out_rec)

    write_jsonl(manifest_out, out_rows)
    print(f"\nWrote {len(out_rows)} rows to: {manifest_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_in", default=Path("data/processed/manifests/ner_manifest.jsonl"), help="Input AMI JSONL manifest")
    parser.add_argument("--manifest_out", required=True, help="Output JSONL with predictions")
    parser.add_argument("--model", default="dslim/bert-base-NER", help="HF token-classification model id")
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU, ...")
    parser.add_argument("--no_text", action="store_true", help="Do not include the original text in output")
    args = parser.parse_args()

    main(
        manifest_in=args.manifest_in,
        manifest_out=args.manifest_out,
        model_name=args.model,
        device=args.device,
        keep_text=not args.no_text
    )