import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import run_subprocess, safe_name


# Leave empty to run all models, or list model IDs to run only those.
RUN_ONLY = [
    # "flair/ner-english-ontonotes-large",
]


def main():
    runner = "scripts/ner_models/bilstm_crf_ner/run_bilstm_crf_ner.py"

    out_dir = Path("data/processed/ner_predictions/bilstm_crf_ner")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        "flair/ner-english-ontonotes-large",
    ]

    if RUN_ONLY:
        models = [m for m in models if m in RUN_ONLY]

    for model_id in models:
        out_file = out_dir / f"{safe_name(model_id)}_predictions.jsonl"
        run_subprocess(
            runner,
            "--manifest_out", str(out_file),
            "--model", model_id,
        )


if __name__ == "__main__":
    main()
