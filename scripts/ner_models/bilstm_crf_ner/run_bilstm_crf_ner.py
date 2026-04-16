import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import iter_jsonl, write_jsonl


def main(
    manifest_in: str,
    manifest_out: str,
    model_name: str = "ner",   # Flair NER tagger (BiLSTM-CRF)
    keep_text: bool = True,
):
    """
    Runs a BiLSTM-CRF sequence tagger (Flair) over each segment's text
    and outputs predicted entity spans as character offsets.

    model_name examples:
      - "ner"       (standard)
      - "ner-fast"  (faster/smaller)
    """
    tagger = SequenceTagger.load(model_name)

    out_rows: List[Dict[str, Any]] = []
    for rec in tqdm(iter_jsonl(manifest_in), desc=f"Flair BiLSTM-CRF NER: {model_name}"):
        text = rec.get("text", "") or ""

        sentence = Sentence(text, use_tokenizer=True)

        t0 = time.perf_counter()
        tagger.predict(sentence)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        predicted_entities = []
        # "ner" is the default tag type for Flair NER models
        for span in sentence.get_spans("ner"):
            # Flair spans carry character offsets in the original sentence text
            start = int(span.start_position)
            end = int(span.end_position)

            predicted_entities.append({
                "start_char": start,
                "end_char": end,
                "label": span.tag,                  # e.g., PER/ORG/LOC/MISC (depends on model)
                "text": text[start:end],
                "score": float(span.score)          # confidence
            })

        out_rec: Dict[str, Any] = {
            "meeting_id": rec.get("meeting_id"),
            "segment_id": rec.get("segment_id"),
            "speaker_id": rec.get("speaker_id"),
            "start_time": rec.get("start_time"),
            "end_time": rec.get("end_time"),
            "overlap": rec.get("overlap"),
            "model": f"flair:{model_name}",
            "latency_ms": latency_ms,
            "predicted_entities": predicted_entities
        }
        if keep_text:
            out_rec["text"] = text

        out_rows.append(out_rec)

    write_jsonl(manifest_out, out_rows)
    print(f"\nWrote {len(out_rows)} rows to: {manifest_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_in", default="data/processed/manifests/ner_manifest.jsonl", help="Input AMI JSONL manifest")
    parser.add_argument("--manifest_out", required=True, help="Output JSONL predictions")
    parser.add_argument("--model", default="ner", help='Flair model: "ner" or "ner-fast"')
    parser.add_argument("--no_text", action="store_true", help="Do not include the original text in output")
    args = parser.parse_args()

    main(
        manifest_in=args.manifest_in,
        manifest_out=args.manifest_out,
        model_name=args.model,
        keep_text=not args.no_text
    )