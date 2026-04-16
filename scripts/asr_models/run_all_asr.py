import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import run_subprocess, safe_name


def main():
    runner = "scripts/asr_models/run_asr.py"

    manifest_in = "data/processed/manifests/asr_manifest_sample20k.jsonl"
    out_dir = Path("data/processed/asr_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"  # "cuda" or "cpu"
    whisper_compute_type = "float16" if device == "cuda" else "int8"

    experiments = [
        # --- Existing ---
        {
            "backend": "whisperx",
            "model": "small",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
                "word_timestamps": True,
            },
        },
        {
            "backend": "whisperx",
            "model": "small",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
            },
        },
        {
            "backend": "conformer_ctc",
            "model": "stt_en_conformer_ctc_large",
            "extra": {
                "overwrite": True,
            },
        },

        # --- Whisper large-v3-turbo (faster large-v3 with ~4x fewer decoder layers) ---
        {
            "backend": "whisper",
            "model": "large-v3-turbo",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
                "word_timestamps": True,
            },
        },
        {
            "backend": "whisper",
            "model": "large-v3-turbo",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
            },
        },

        # --- Distil-Whisper large-v3 (~6x faster distillation of large-v3) ---
        {
            "backend": "whisper",
            "model": "distil-large-v3",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
                "word_timestamps": True,
            },
        },
        {
            "backend": "whisper",
            "model": "distil-large-v3",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
            },
        },

        # --- Parakeet TDT 1.1B (NVIDIA transducer, strong on conversational speech) ---
        {
            "backend": "canary",
            "model": "nvidia/parakeet-tdt-1.1b",
            "extra": {
                "overwrite": True,
            },
        },

        # --- Parakeet CTC 1.1B (NVIDIA CTC variant, faster decoding) ---
        {
            "backend": "canary",
            "model": "nvidia/parakeet-ctc-1.1b",
            "extra": {
                "overwrite": True,
            },
        },

        # --- WhisperX large-v3-turbo (forced alignment gives better timestamps than built-in) ---
        {
            "backend": "whisperx",
            "model": "large-v3-turbo",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
                "word_timestamps": True,
            },
        },

        # --- WhisperX distil-large-v3 (fastest option with forced alignment) ---
        {
            "backend": "whisperx",
            "model": "distil-large-v3",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
                "word_timestamps": True,
            },
        },

        # --- Whisper medium.en (English-only; better WER than multilingual medium on English data) ---
        {
            "backend": "whisper",
            "model": "medium.en",
            "extra": {
                "compute_type": whisper_compute_type,
                "language": "en",
                "overwrite": True,
                "word_timestamps": True,
            },
        },
    ]

    for exp in experiments:
        backend = exp["backend"]
        model = exp["model"]
        extra = exp.get("extra", {})

        suffix_parts = [backend, model]
        if extra.get("word_timestamps"):
            suffix_parts.append("word_timestamps")
        out_file = out_dir / f"{safe_name('__'.join(suffix_parts))}_predictions.jsonl"

        cmd_args = [
            "--manifest_in", manifest_in,
            "--manifest_out", str(out_file),
            "--backend", backend,
            "--model", model,
            "--device", device,
        ]

        for k, v in extra.items():
            flag = f"--{k}"
            if isinstance(v, bool):
                if v:
                    cmd_args.append(flag)
            else:
                cmd_args.extend([flag, str(v)])

        run_subprocess(runner, *cmd_args)


if __name__ == "__main__":
    main()
