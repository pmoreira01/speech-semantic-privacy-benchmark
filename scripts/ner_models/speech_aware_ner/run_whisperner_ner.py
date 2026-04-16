import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import iter_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

# Cache resamplers so they are not re-created for every segment
_resamplers: Dict[int, torchaudio.transforms.Resample] = {}


def find_audio_file(audio_root: Path, meeting_id: str, audio_suffix: str) -> Optional[Path]:
    matches = list(audio_root.rglob(f"{meeting_id}{audio_suffix}"))
    return matches[0] if matches else None


def load_audio_segment(
    audio_path: Path, start_time: float, end_time: float, target_sr: int = 16000
) -> np.ndarray:
    signal, sr = torchaudio.load(str(audio_path))

    if sr != target_sr:
        if sr not in _resamplers:
            _resamplers[sr] = torchaudio.transforms.Resample(sr, target_sr)
        signal = _resamplers[sr](signal)

    if signal.ndim == 2:
        signal = signal.mean(dim=0)

    start_sample = max(0, int(start_time * target_sr))
    end_sample = min(signal.shape[-1], int(end_time * target_sr))
    return signal[start_sample:end_sample].cpu().numpy()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_whisperner_output(text: str) -> List[Dict[str, Any]]:
    """Parse WhisperNER tagged output: <person>David<person>"""
    entities = []
    pattern = re.compile(r"<([^>\s]+)>(.*?)<\1>", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        label = match.group(1).strip()
        ent_text = match.group(2).strip()
        if ent_text:
            entities.append({"label": label, "text": ent_text})
    return entities


def align_entities_to_text(text: str, parsed_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find entity substrings in plain text and attach char offsets."""
    aligned = []
    used: set = set()
    for ent in parsed_entities:
        ent_text = ent["text"]
        label = ent["label"]
        start_search = 0
        while True:
            start = text.find(ent_text, start_search)
            if start == -1:
                break
            end = start + len(ent_text)
            key = (start, end, label)
            if key not in used:
                used.add(key)
                aligned.append({
                    "text":       ent_text,
                    "label":      label.upper(),
                    "start_char": start,
                    "end_char":   end,
                    "score":      1.0,
                })
                break
            start_search = start + 1
    return aligned


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def process_batch(
    batch: List[Dict],
    audio_cache: Dict[str, Optional[Path]],
    audio_root: Path,
    audio_suffix: str,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    prompt_ids: torch.Tensor,
    language: str,
    device: torch.device,
    keep_text: bool,
    model_name: str,
) -> List[Dict]:
    # Build output records and load audio for each item in the batch
    out_recs = []
    audio_segments = []
    valid_indices = []

    for rec in batch:
        meeting_id = rec.get("meeting_id")
        start_time = float(rec.get("start_time", 0.0))
        end_time = float(rec.get("end_time", 0.0))

        # Cached filesystem lookup — rglob is expensive when called per-segment
        if meeting_id not in audio_cache:
            audio_cache[meeting_id] = find_audio_file(audio_root, meeting_id, audio_suffix)
        audio_path = audio_cache[meeting_id]

        out_rec = {
            "meeting_id":        meeting_id,
            "segment_id":        rec.get("segment_id"),
            "speaker_id":        rec.get("speaker_id"),
            "start_time":        start_time,
            "end_time":          end_time,
            "overlap":           rec.get("overlap"),
            "model":             model_name,
            "audio_file":        str(audio_path) if audio_path else None,
            "asr_ner_text":      "",
            "latency_ms":        None,
            "predicted_entities": [],
        }
        if keep_text:
            out_rec["text"] = rec.get("text", "")

        if audio_path is None:
            out_rec["error"] = f"Audio file not found for meeting_id={meeting_id}"
            out_recs.append(out_rec)
            audio_segments.append(None)
            continue

        try:
            seg = load_audio_segment(audio_path, start_time, end_time)
            if seg.size == 0:
                raise ValueError("Empty audio segment")
            audio_segments.append(seg)
            valid_indices.append(len(out_recs))
        except Exception as e:
            out_rec["error"] = str(e)
            audio_segments.append(None)

        out_recs.append(out_rec)

    if not valid_indices:
        return out_recs

    # Batch inference — Whisper always pads to 30s so all feature tensors are the same shape
    valid_audio = [audio_segments[i] for i in valid_indices]
    try:
        features = processor(
            valid_audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            predicted_ids = model.generate(
                features,
                prompt_ids=prompt_ids,
                generation_config=model.generation_config,
                language=language,
            )
        latency_per_seg = round((time.perf_counter() - t0) * 1000.0 / len(valid_indices), 3)

        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        for local_i, global_i in enumerate(valid_indices):
            raw_output = decoded[local_i].strip()
            parsed = parse_whisperner_output(raw_output)
            plain_text = re.sub(r"<\s*/?\s*[^>]+>", "", raw_output).strip()
            aligned = align_entities_to_text(plain_text, parsed)

            out_recs[global_i]["asr_ner_text"] = raw_output
            out_recs[global_i]["latency_ms"] = latency_per_seg
            out_recs[global_i]["predicted_entities"] = aligned

    except Exception as e:
        for i in valid_indices:
            out_recs[i]["error"] = str(e)

    return out_recs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_in",   default="data/processed/manifests/ner_manifest_sample2k.jsonl")
    parser.add_argument("--manifest_out",  required=True)
    parser.add_argument("--audio_root",    required=True)
    parser.add_argument("--audio_suffix",  default=".Mix-Headset.wav")
    parser.add_argument("--model",         default="aiola/whisper-ner-v1")
    parser.add_argument("--prompt",        default="person, company, location, date, time, money")
    parser.add_argument("--language",      default="en")
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_text",       action="store_true")
    parser.add_argument("--overwrite",     action="store_true")
    args = parser.parse_args()

    manifest_out = Path(args.manifest_out)
    if manifest_out.exists():
        if args.overwrite:
            manifest_out.unlink()
        else:
            raise FileExistsError(f"{manifest_out} already exists. Use --overwrite to replace it.")

    device = torch.device(args.device)
    print(f"Loading model: {args.model} on {device}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    prompt_ids = processor.get_prompt_ids(args.prompt.lower(), return_tensors="pt").to(device)

    # Warmup — avoids cold-start CUDA init inflating the first batch's latency
    print("Warming up...")
    dummy = processor(
        [np.zeros(16000, dtype=np.float32)],
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)
    with torch.no_grad():
        _ = model.generate(
            dummy,
            prompt_ids=prompt_ids,
            generation_config=model.generation_config,
            language=args.language,
            max_new_tokens=5,
        )

    model_name = f"whisperner:{args.model}"
    audio_cache: Dict[str, Optional[Path]] = {}
    audio_root = Path(args.audio_root)

    all_records = list(iter_jsonl(args.manifest_in))
    out_rows = []
    buffer = []

    for rec in tqdm(all_records, desc=f"WhisperNER ({args.model})", total=len(all_records)):
        buffer.append(rec)
        if len(buffer) >= args.batch_size:
            out_rows.extend(process_batch(
                buffer, audio_cache, audio_root, args.audio_suffix,
                processor, model, prompt_ids, args.language, device,
                keep_text=not args.no_text, model_name=model_name,
            ))
            buffer = []

    if buffer:
        out_rows.extend(process_batch(
            buffer, audio_cache, audio_root, args.audio_suffix,
            processor, model, prompt_ids, args.language, device,
            keep_text=not args.no_text, model_name=model_name,
        ))

    write_jsonl(manifest_out, out_rows)
    print(f"\nWrote {len(out_rows)} rows -> {manifest_out}")


if __name__ == "__main__":
    main()
