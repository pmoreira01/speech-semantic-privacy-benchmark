import subprocess
import sys
from pathlib import Path


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, script] + list(args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace("+", "_").replace("__", "_")


def main():
    runner = "scripts/asr_models/run_asr.py"

    manifest_in = "data/processed/manifests/asr_manifest_sample20k.jsonl"
    out_dir = Path("data/processed/asr_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Adjust for your machine
    device = "cuda"  # "cuda" or "cpu"

    whisper_compute_type = "float16" if device == "cuda" else "int8"

    experiments = [
        # Whisper variants without word timestamps
        #{
        #    "backend": "whisper",
        #    "model": "tiny",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "base",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "small",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "medium",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "large-v3",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "overwrite": True,
        #    },
        #},

        # Whisper variants with word timestamps
        #{
        #    "backend": "whisper",
        #    "model": "tiny",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "word_timestamps": True,
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "base",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "word_timestamps": True,
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "small",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "word_timestamps": True,
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "medium",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "word_timestamps": True,
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "whisper",
        #    "model": "large-v3",
        #    "extra": {
        #        "compute_type": whisper_compute_type,
        #        "language": "en",
        #        "word_timestamps": True,
        #        "overwrite": True,
        #    },
        #},

        # Other backends supported by the updated script
        #{
        #    "backend": "speechbrain",
        #    "model": "speechbrain/asr-crdnn-rnnlm-librispeech",
        #    "extra": {
        #        "overwrite": True,
        #    },
        #},
        #{
        #    "backend": "wav2vec2",
        #    "model": "facebook/wav2vec2-large-960h-lv60-self",
        #    "extra": {
        #        "overwrite": True,
        #    },
        #},

        # Only keep this if run_asr.py supports "canary"
        #{
        #     "backend": "canary",
        #     "model": "nvidia/canary-1b-v2",
        #     "extra": {
        #         "overwrite": True,
        #     },
        #},
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

        run(runner, *cmd_args)


if __name__ == "__main__":
    main()