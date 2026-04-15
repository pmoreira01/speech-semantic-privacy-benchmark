import subprocess
import sys
from pathlib import Path


def run(script, *args):
    cmd = [sys.executable, script] + list(args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def main():

    runner = "scripts/ner_models/bilstm_crf_ner/run_bilstm_crf_ner.py"

    out_dir = Path("data/processed/ner_predictions/bilstm_crf_ner")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        #"ner",
        #"ner-fast",
        #"ner-large",
        #"flair/ner-english-ontonotes",
        #"flair/ner-english-large",
        "flair/ner-english-ontonotes-large",
    ]

    for model_id in models:

        out_file = out_dir / f"{safe_name(model_id)}_predictions.jsonl"

        run(
            runner,
            "--manifest_out", str(out_file),
            "--model", model_id
        )


if __name__ == "__main__":
    main()