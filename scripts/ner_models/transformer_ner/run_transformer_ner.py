import json
import time
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm
from transformers import pipeline


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_bio(label: str) -> str:
    if label.startswith(("B-", "I-")):
        return label[2:]
    return label


def preds_to_entities(text: str, preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    predicted_entities = []
    for p in preds:
        start = int(p["start"])
        end = int(p["end"])
        raw_label = p.get("entity_group") or p.get("entity") or "MISC"
        label = strip_bio(str(raw_label))
        score = float(p.get("score", 0.0))
        predicted_entities.append(
            {
                "start_char": start,
                "end_char": end,
                "label": label,
                "text": text[start:end],
                "score": score,
            }
        )
    return predicted_entities


def main(
    manifest_in: str,
    manifest_out: str,
    model_name: str = "dslim/bert-base-NER",
    device: int = -1,
    keep_text: bool = True,
    batch_size: int = 32,
):
    ner = pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device=device,
    )

    # Warmup (helps stabilize GPU timings + avoids first-call spike)
    _ = ner("Warmup sentence for NER.", batch_size=1)

    out_rows: List[Dict[str, Any]] = []

    buffer: List[Dict[str, Any]] = []
    total = 0

    def flush(buf: List[Dict[str, Any]]):
        nonlocal total
        if not buf:
            return

        texts = [(r.get("text", "") or "") for r in buf]

        t0 = time.perf_counter()
        batch_preds = ner(texts, batch_size=len(texts))
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = batch_latency_ms / max(1, len(texts))

        for rec, text, preds in zip(buf, texts, batch_preds):
            out_rec = {
                "meeting_id": rec.get("meeting_id"),
                "segment_id": rec.get("segment_id"),
                "speaker_id": rec.get("speaker_id"),
                "start_time": rec.get("start_time"),
                "end_time": rec.get("end_time"),
                "overlap": rec.get("overlap"),
                "model": model_name,
                # estimated per-utterance time (useful for comparing throughput)
                "latency_ms": per_item_ms,
                # optional batch diagnostics
                "batch_latency_ms": batch_latency_ms,
                "batch_size": len(texts),
                "predicted_entities": preds_to_entities(text, preds),
            }
            if keep_text:
                out_rec["text"] = text

            out_rows.append(out_rec)
            total += 1

    # Iterate with progress bar (unknown length -> no %; still shows count/s)
    for rec in tqdm(iter_jsonl(manifest_in), desc=f"NER (batched): {model_name}"):
        buffer.append(rec)
        if len(buffer) >= batch_size:
            flush(buffer)
            buffer = []

    flush(buffer)

    write_jsonl(manifest_out, out_rows)
    print(f"\nWrote {len(out_rows)} rows to: {manifest_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_in",
        default="data/processed/manifests/ner_manifest.jsonl",
        help="Input AMI JSONL manifest",
    )
    parser.add_argument("--manifest_out", required=True, help="Output JSONL with predictions")
    parser.add_argument("--model", default="dslim/bert-base-NER", help="HF token-classification model id")
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 GPU, ...")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for pipeline inference")
    parser.add_argument("--no_text", action="store_true", help="Do not include the original text in output")
    args = parser.parse_args()

    main(
        manifest_in=args.manifest_in,
        manifest_out=args.manifest_out,
        model_name=args.model,
        device=args.device,
        keep_text=not args.no_text,
        batch_size=args.batch_size,
    )