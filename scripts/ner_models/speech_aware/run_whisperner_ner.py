import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration


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


def find_audio_file(audio_root: Path, meeting_id: str, audio_suffix: str) -> Optional[Path]:
    matches = list(audio_root.rglob(f"{meeting_id}{audio_suffix}"))
    return matches[0] if matches else None


def load_audio_segment(audio_path: Path, start_time: float, end_time: float, target_sr: int = 16000):
    signal, sr = torchaudio.load(str(audio_path))

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        signal = resampler(signal)
        sr = target_sr

    if signal.ndim == 2:
        signal = torch.mean(signal, dim=0)

    start_sample = max(0, int(start_time * sr))
    end_sample = min(signal.shape[-1], int(end_time * sr))
    segment = signal[start_sample:end_sample]

    return segment, sr


def parse_whisperner_output(text: str) -> List[Dict[str, Any]]:
    """
    Best-effort parser for tagged WhisperNER output.
    Adjust this if you inspect a few outputs and find a different tag format.
    Expected likely patterns:
      <person> David </person>
      <company> Microsoft </company>
    """
    entities = []

    # pattern like <person>David</person> or <person> David </person>
    pattern = re.compile(r"<\s*([^>\s]+)\s*>(.*?)<\s*/\s*\1\s*>", flags=re.IGNORECASE | re.DOTALL)

    for m in pattern.finditer(text):
        label = m.group(1).strip()
        ent_text = m.group(2).strip()
        if ent_text:
            entities.append({
                "label": label,
                "text": ent_text
            })

    return entities


def align_entities_to_text(text: str, parsed_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find entity substrings in the final output text.
    This is best-effort and works well when entity text appears exactly once.
    """
    aligned = []
    used = set()

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
                    "start_char": start,
                    "end_char": end,
                    "label": label,
                    "text": ent_text
                })
                break
            start_search = start + 1

    return aligned


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest_in", default="data/processed/manifests/ner_manifest_sample2k.jsonl")
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--audio_suffix", default=".Mix-Headset.wav")

    parser.add_argument("--model", default="aiola/whisper-ner-v1")
    parser.add_argument("--prompt", default="person, company, location, date, time, money")
    parser.add_argument("--language", default="en")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_text", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    audio_root = Path(args.audio_root)

    out_rows = []

    for rec in tqdm(iter_jsonl(args.manifest_in), desc="WhisperNER"):
        meeting_id = rec.get("meeting_id")
        start_time = float(rec.get("start_time", 0.0))
        end_time = float(rec.get("end_time", 0.0))

        audio_path = find_audio_file(audio_root, meeting_id, args.audio_suffix)

        out_rec = {
            "meeting_id": meeting_id,
            "segment_id": rec.get("segment_id"),
            "speaker_id": rec.get("speaker_id"),
            "start_time": start_time,
            "end_time": end_time,
            "overlap": rec.get("overlap"),
            "model": f"whisperner:{args.model}",
            "audio_file": str(audio_path) if audio_path else None,
            "prompt": args.prompt,
            "asr_ner_text": "",
            "latency_ms": None,
            "predicted_entities": [],
        }

        if not args.no_text:
            out_rec["text"] = rec.get("text", "")

        if audio_path is None:
            out_rec["error"] = f"Audio file not found for meeting_id={meeting_id}"
            out_rows.append(out_rec)
            continue

        try:
            audio_segment, sr = load_audio_segment(audio_path, start_time, end_time, target_sr=16000)

            if audio_segment.numel() == 0:
                out_rec["error"] = "Empty audio segment"
                out_rows.append(out_rec)
                continue

            input_features = processor(
                audio_segment.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)

            prompt_ids = processor.get_prompt_ids(args.prompt.lower(), return_tensors="pt")
            prompt_ids = prompt_ids.to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    prompt_ids=prompt_ids,
                    generation_config=model.generation_config,
                    language=args.language,
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            raw_output = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            # Best-effort parse of tagged entities from the generated text
            parsed_entities = parse_whisperner_output(raw_output)

            # Remove tags for a plain text version
            plain_text = re.sub(r"<\s*/?\s*[^>]+>", "", raw_output).strip()
            aligned_entities = align_entities_to_text(plain_text, parsed_entities)

            out_rec["asr_ner_text"] = raw_output
            out_rec["latency_ms"] = latency_ms
            out_rec["predicted_entities"] = aligned_entities

        except Exception as e:
            out_rec["error"] = str(e)

        out_rows.append(out_rec)

    write_jsonl(args.manifest_out, out_rows)
    print(f"\nWrote {len(out_rows)} rows to: {args.manifest_out}")


if __name__ == "__main__":
    main()