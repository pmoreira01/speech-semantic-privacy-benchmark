import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    from scipy.signal import resample_poly

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def load_audio_segment(
    audio_path: Path,
    start_time: float,
    end_time: float,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    if end_time <= start_time:
        return np.array([], dtype=np.float32), target_sr

    with sf.SoundFile(str(audio_path), "r") as f:
        sr = int(f.samplerate)
        total_frames = len(f)

        start_frame = max(0, int(start_time * sr))
        end_frame = min(total_frames, int(end_time * sr))

        if end_frame <= start_frame:
            return np.array([], dtype=np.float32), target_sr

        frames_to_read = end_frame - start_frame
        f.seek(start_frame)
        segment = f.read(frames=frames_to_read, dtype="float32", always_2d=False)

    if segment.size == 0:
        return np.array([], dtype=np.float32), target_sr

    if segment.ndim > 1:
        segment = segment.mean(axis=1)

    segment = np.asarray(segment, dtype=np.float32)
    segment = resample_audio(segment, sr, target_sr)
    return segment, target_sr


def build_backend(args):
    backend = args.backend.lower()

    if backend == "whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

        def transcribe_fn(audio_segment: np.ndarray, sr: int) -> Dict[str, Any]:
            segments, _ = model.transcribe(
                audio_segment,
                language=args.language,
                beam_size=1,
                vad_filter=False,
                word_timestamps=args.word_timestamps,
            )

            texts = []
            words_out = []

            for seg in segments:
                seg_text = (seg.text or "").strip()
                if seg_text:
                    texts.append(seg_text)

                if args.word_timestamps and getattr(seg, "words", None):
                    for w in seg.words:
                        word_text = getattr(w, "word", None)
                        if word_text is None:
                            continue
                        words_out.append(
                            {
                                "word": word_text,
                                "start": getattr(w, "start", None),
                                "end": getattr(w, "end", None),
                                "probability": getattr(w, "probability", None),
                            }
                        )

            result = {"text": " ".join(texts).strip()}
            if args.word_timestamps:
                result["words"] = words_out
            return result

        return transcribe_fn

    if backend == "wav2vec2":
        from transformers import pipeline

        hf_device = 0 if args.device == "cuda" else -1
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model,
            device=hf_device,
        )

        def transcribe_fn(audio_segment: np.ndarray, sr: int) -> Dict[str, Any]:
            result = asr_pipe({"array": audio_segment, "sampling_rate": sr})
            return {"text": result["text"].strip()}

        return transcribe_fn

    if backend == "speechbrain":
        from speechbrain.inference.ASR import EncoderDecoderASR

        savedir = args.cache_dir or "pretrained_asr_speechbrain"
        run_opts = {"device": args.device}
        asr_model = EncoderDecoderASR.from_hparams(
            source=args.model,
            savedir=savedir,
            run_opts=run_opts,
        )

        def transcribe_fn(audio_segment: np.ndarray, sr: int) -> Dict[str, Any]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_segment, sr)
                text = asr_model.transcribe_file(tmp.name).strip()
            return {"text": text}

        return transcribe_fn

    raise ValueError(f"Unsupported backend: {args.backend}")


def build_output_record(rec: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    return {
        "meeting_id": rec.get("meeting_id"),
        "segment_id": rec.get("segment_id"),
        "speaker_id": rec.get("speaker_id"),
        "start_time": rec.get("start_time"),
        "end_time": rec.get("end_time"),
        "duration": rec.get("duration"),
        "audio": rec.get("audio"),
        "reference_text": rec.get("reference_text", ""),
        "overlap": rec.get("overlap"),
        "model": model_name,
        "asr_text": "",
        "latency_ms": None,
    }


def extract_audio_path(rec: Dict[str, Any]) -> Path:
    audio_info = rec.get("audio")
    if not isinstance(audio_info, dict):
        raise ValueError("Missing or invalid 'audio' object")
    audio_path = audio_info.get("path")
    if not audio_path:
        raise ValueError("Missing 'audio.path'")
    return Path(audio_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_in", required=True)
    parser.add_argument("--manifest_out", required=True)

    parser.add_argument("--backend", required=True, choices=["whisper", "wav2vec2", "speechbrain"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="int8")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--target_sr", type=int, default=16000)

    # Whisper-specific
    parser.add_argument(
        "--word_timestamps",
        action="store_true",
        help="Enable word-level timestamps for Whisper backend.",
    )

    # Optional convenience
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output manifest if it already exists.",
    )

    args = parser.parse_args()

    if args.word_timestamps and args.backend != "whisper":
        raise ValueError("--word_timestamps is only supported with --backend whisper")

    manifest_in = Path(args.manifest_in)
    manifest_out = Path(args.manifest_out)

    if not manifest_in.exists():
        raise FileNotFoundError(f"Input manifest not found: {manifest_in}")

    if manifest_out.exists():
        if args.overwrite:
            manifest_out.unlink()
        else:
            raise FileExistsError(
                f"Output manifest already exists: {manifest_out}. "
                f"Use --overwrite to replace it."
            )

    transcribe_fn = build_backend(args)
    model_name = f"{args.backend}:{args.model}"

    total = count_jsonl(manifest_in)

    written = 0
    with tqdm(total=total, desc=f"ASR: {model_name}") as pbar:
        for rec in iter_jsonl(manifest_in):
            out_rec = build_output_record(rec, model_name)

            try:
                audio_path = extract_audio_path(rec)

                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                start_time = float(rec["start_time"])
                end_time = float(rec["end_time"])

                audio_segment, sr = load_audio_segment(
                    audio_path=audio_path,
                    start_time=start_time,
                    end_time=end_time,
                    target_sr=args.target_sr,
                )

                if len(audio_segment) == 0:
                    raise ValueError("Empty audio segment")

                t0 = time.perf_counter()
                asr_result = transcribe_fn(audio_segment, sr)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                out_rec["asr_text"] = asr_result.get("text", "")
                out_rec["latency_ms"] = latency_ms

                if args.backend == "whisper" and args.word_timestamps:
                    out_rec["word_timestamps"] = asr_result.get("words", [])

            except Exception as e:
                out_rec["error"] = str(e)

            append_jsonl(manifest_out, out_rec)
            written += 1
            pbar.update(1)

    print(f"Wrote {written} rows -> {manifest_out}")


if __name__ == "__main__":
    main()