import argparse
import subprocess
import sys
from pathlib import Path


def safe_name(model_id: str) -> str:
    # "llama3.1:8b" -> "llama3.1_8b"
    return model_id.replace("/", "_").replace(":", "_")


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed/ner_predictions/llm_prompted_ner")
    ap.add_argument("--ollama_url", default="http://localhost:11434/api/generate")
    ap.add_argument("--device_note", default="local", help="Optional note only for filenames/logging")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--continue_on_fail", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_ctx", type=int, default=None)
    ap.add_argument("--timeout_s", type=int, default=30)
    ap.add_argument("--max_chars", type=int, default=1200)
    args = ap.parse_args()

    runner = "scripts/ner_models/llm_prompted_ner/run_llm_ner.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A reasonable local LLM coverage set (edit to match what you pulled in Ollama)
    models = [
        #"llama3.1:8b",
        #"mistral:7b",
        #"qwen2.5:7b",
        #"gemma2:9b",
        #"phi3:mini",
        "llama3.2:3b",
        "smollm2",
        "tinyllama",
    ]

    for m in models:
        out_file = out_dir / f"{safe_name(m)}_predictions.jsonl"
        if args.skip_existing and out_file.exists():
            print(f"Skipping existing: {out_file}")
            continue

        cmd = [
            sys.executable, runner,
            "--manifest_out", str(out_file),
            "--model", m,
            "--ollama_url", args.ollama_url,
            "--temperature", str(args.temperature),
            "--timeout_s", str(args.timeout_s),
            "--max_chars", str(args.max_chars),
            #"--keep_raw"
        ]
        if args.num_ctx is not None:
            cmd += ["--num_ctx", str(args.num_ctx)]

        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            if args.continue_on_fail:
                print(f"❌ Model failed: {m} (continuing). Error: {e}")
                continue
            raise


if __name__ == "__main__":
    main()