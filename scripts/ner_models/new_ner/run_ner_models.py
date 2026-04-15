"""
Open-source NER benchmark runner — GLiNER, HydroX, and Presidio backends.

Input manifest:
  {"meeting_id": ..., "segment_id": ..., "speaker_id": ...,
   "start_time": ..., "end_time": ..., "overlap": ..., "text": "..."}

Output manifest adds:
  {"model": ..., "latency_ms": ..., "predicted_entities": [...], ...}

Entity format (consistent across all backends):
  {"text": "John Smith", "label": "PERSON", "start_char": 4, "end_char": 14, "score": 0.98}

Usage:
  # GLiNER (medium)
  python run_ner_models.py --backend gliner --model urchade/gliner_medium-v2.1 --manifest_out results/gliner.jsonl

  # NuNER (GLiNER variant)
  python run_ner_models.py --backend gliner --model numind/NuNER-zero --manifest_out results/nuner.jsonl

  # Custom GLiNER labels
  python run_ner_models.py --backend gliner --gliner_labels '["person name","phone number","SSN"]' --manifest_out results/gliner_custom.jsonl

  # HydroX PII Masker
  python run_ner_models.py --backend hydrox --manifest_out results/hydrox.jsonl

  # Presidio + spaCy
  python run_ner_models.py --backend presidio --manifest_out results/presidio.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    return device


# ---------------------------------------------------------------------------
# Manifest I/O (unchanged from your original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared entity format
# ---------------------------------------------------------------------------

def make_entity(text: str, label: str, start: int, end: int, score: float = 1.0) -> Dict:
    return {"text": text, "label": label.upper(), "start_char": start, "end_char": end, "score": round(score, 4)}


# ---------------------------------------------------------------------------
# Backend: GLiNER  (zero-shot, labels defined at runtime)
# ---------------------------------------------------------------------------

# Default PII-focused label set — edit freely
GLINER_DEFAULT_LABELS = [
    "person name",
    "phone number",
    "email address",
    "social security number",
    "credit card number",
    "bank account number",
    "physical address",
    "date of birth",
    "organization",
    "location",
    "ip address",
    "passport number",
    "driver license number",
]


class GLiNERBackend:
    """GLiNER / NuNER-zero — zero-shot NER, processes one utterance at a time."""

    def __init__(self, model_name: str, labels: Optional[List[str]] = None, threshold: float = 0.5, device: str = "cpu"):
        from gliner import GLiNER
        self.model_name = model_name
        self.labels = labels or GLINER_DEFAULT_LABELS
        self.threshold = threshold
        print(f"Loading GLiNER model: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)
        self.model = self.model.to(device)
        # Warmup
        _ = self.model.predict_entities("Warmup.", self.labels)

    def run_batch(self, texts: List[str]) -> tuple[List[List[Dict]], List[float]]:
        results = []
        latencies_ms = []
        for text in texts:
            t0 = time.perf_counter()
            preds = self.model.predict_entities(text, self.labels, threshold=self.threshold)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            results.append([
                make_entity(
                    text=p["text"],
                    label=p["label"],
                    start=p["start"],
                    end=p["end"],
                    score=p.get("score", 1.0),
                )
                for p in preds
            ])
        return results, latencies_ms


# ---------------------------------------------------------------------------
# Backend: HydroX PII Masker  (DeBERTa-v3)
# ---------------------------------------------------------------------------

class HydroXBackend:
    """
    HydroX PII Masker — pip install pii-masker
    Processes one utterance at a time; returns entity spans from the masker dict.
    """

    def __init__(self, device: str = "cpu"):
        from pii_masker import CustomPIIMasker
        self.model_name = "hydrox/pii-masker"
        print("Loading HydroX PII Masker...")
        self.masker = CustomPIIMasker()
        _ = self.masker.get_detected_entities("Warmup sentence with John Doe.")

    def run_batch(self, texts: List[str]) -> tuple[List[List[Dict]], List[float]]:
        results = []
        latencies_ms = []
        for text in texts:
            t0 = time.perf_counter()
            preds = self.masker.get_detected_entities(text)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            entities = [
                make_entity(
                    text=p["text"],
                    label=p["entity_type"],
                    start=p["start"],
                    end=p["end"],
                    score=p["score"],
                )
                for p in preds
            ]
            results.append(entities)
        return results, latencies_ms


# ---------------------------------------------------------------------------
# Backend: Microsoft Presidio + spaCy
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Backend: Microsoft Presidio + spaCy
# ---------------------------------------------------------------------------
 
# Entity types Presidio supports out of the box — add/remove as needed
PRESIDIO_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "URL",
    "IBAN_CODE",
    "MEDICAL_LICENSE",
    "ORG",
    "NRP",          # Presidio's type for nationalities/religions/political groups — maps to ORG
]
 
 
class PresidioBackend:
    """
    Presidio + spaCy — pip install presidio-analyzer presidio-anonymizer
                        python -m spacy download en_core_web_lg
    Combines ML (spaCy) with regex recognizers for pattern-based PII.
    """
 
    def __init__(self, entities: Optional[List[str]] = None):
        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        spacy.require_gpu()
        self.model_name = "presidio+spacy(en_core_web_trf)"
        self.entities = entities or PRESIDIO_ENTITIES
        print("Loading Presidio analyzer (en_core_web_trf)...")
        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}],
        }
        provider = NlpEngineProvider(nlp_configuration=config)
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        # Warmup
        _ = self.analyzer.analyze(text="Warmup.", language="en", entities=self.entities)
 
    def run_batch(self, texts: List[str]) -> tuple[List[List[Dict]], List[float]]:
        results = []
        latencies_ms = []
        for text in texts:
            t0 = time.perf_counter()
            recognizer_results = self.analyzer.analyze(
                text=text, language="en", entities=self.entities
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            entities = [
                make_entity(
                    text=text[r.start:r.end],
                    label=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                )
                for r in recognizer_results
            ]
            results.append(entities)
        return results, latencies_ms


# ---------------------------------------------------------------------------
# Generic batch processor (same structure as your original flush() logic)
# ---------------------------------------------------------------------------

def process_manifest(
    backend,
    manifest_in: str,
    manifest_out: str,
    batch_size: int,
    keep_text: bool,
):
    out_rows: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []

    def flush(buf: List[Dict[str, Any]]):
        if not buf:
            return
        texts = [(r.get("text", "") or "") for r in buf]
        batch_entities, latencies_ms = backend.run_batch(texts)

        for rec, entities, latency_ms in zip(buf, batch_entities, latencies_ms):
            out_rec = {
                "meeting_id": rec.get("meeting_id"),
                "segment_id": rec.get("segment_id"),
                "speaker_id": rec.get("speaker_id"),
                "start_time": rec.get("start_time"),
                "end_time": rec.get("end_time"),
                "overlap": rec.get("overlap"),
                "model": backend.model_name,
                "latency_ms": round(latency_ms, 3),
                "predicted_entities": entities,
            }
            if keep_text:
                out_rec["text"] = rec.get("text", "")
            out_rows.append(out_rec)

    for rec in tqdm(iter_jsonl(manifest_in), desc=f"NER: {backend.model_name}"):
        buffer.append(rec)
        if len(buffer) >= batch_size:
            flush(buffer)
            buffer = []

    flush(buffer)
    write_jsonl(manifest_out, out_rows)
    print(f"\nWrote {len(out_rows)} rows → {manifest_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_backend(args) -> Any:
    device = get_device()
    if args.backend == "gliner":
        labels = json.loads(args.gliner_labels) if args.gliner_labels else None
        return GLiNERBackend(
            model_name=args.model or "urchade/gliner_medium-v2.1",
            labels=labels,
            threshold=args.gliner_threshold,
            device=device,
        )
    elif args.backend == "hydrox":
        return HydroXBackend(device=device)
    elif args.backend == "presidio":
        entities = json.loads(args.presidio_entities) if args.presidio_entities else None
        return PresidioBackend(entities=entities)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


def main():
    parser = argparse.ArgumentParser(description="Open-source NER benchmark runner")
    parser.add_argument("--manifest_in", default="data/processed/manifests/ner_manifest.jsonl")
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument(
        "--backend", choices=["gliner", "hydrox", "presidio"], required=True,
        help="Which NER backend to use",
    )

    # Shared
    parser.add_argument("--model", default=None, help="Model name/path (gliner backend only)")
    parser.add_argument("--batch_size", type=int, default=32, help="Buffer size for progress tracking")
    parser.add_argument("--no_text", action="store_true", help="Omit original text from output")

    # GLiNER-specific
    parser.add_argument(
        "--gliner_labels", default=None,
        help='JSON list of entity labels, e.g. \'["person name","phone number"]\'. Defaults to built-in PII set.',
    )
    parser.add_argument("--gliner_threshold", type=float, default=0.4, help="GLiNER confidence threshold")

    # Presidio-specific
    parser.add_argument(
        "--presidio_entities", default=None,
        help='JSON list of Presidio entity types to detect. Defaults to built-in PII set.',
    )

    args = parser.parse_args()
    backend = build_backend(args)

    process_manifest(
        backend=backend,
        manifest_in=args.manifest_in,
        manifest_out=args.manifest_out,
        batch_size=args.batch_size,
        keep_text=not args.no_text,
    )


if __name__ == "__main__":
    main()