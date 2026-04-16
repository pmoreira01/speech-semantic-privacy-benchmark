import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import run_subprocess, safe_name


# Leave empty to run all models, or list model IDs to run only those.
RUN_ONLY = [
    # "dslim/bert-base-NER",
    # "Jean-Baptiste/roberta-large-ner-english",
]


def main():
    runner = "scripts/ner_models/transformer_ner/run_transformer_ner.py"

    out_dir = Path("data/processed/ner_predictions/transformer_ner")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "0"  # set to "-1" for CPU

    models = [
        "dslim/bert-base-NER",
        "Jean-Baptiste/roberta-large-ner-english",
        "elastic/distilbert-base-cased-finetuned-conll03-english",
        "Gladiator/microsoft-deberta-v3-large_ner_conll2003",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "nickprock/bert-finetuned-ner-ontonotes",
        "nickprock/distilbert-finetuned-ner-ontonotes",
    ]

    if RUN_ONLY:
        models = [m for m in models if m in RUN_ONLY]

    for model_id in models:
        out_file = out_dir / f"{safe_name(model_id)}_predictions.jsonl"
        run_subprocess(
            runner,
            "--manifest_out", str(out_file),
            "--model", model_id,
            "--device", device,
        )


if __name__ == "__main__":
    main()
