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
    runner = "scripts/ner_models/transformer_ner/run_transformer_ner.py"
    
    out_dir = Path("data/processed/ner_predictions/transformer_ner")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "0"  # set to "-1" for CPU

    models = [
        # Baseline
        "dslim/bert-base-NER",

        # Strong encoder
        "Jean-Baptiste/roberta-large-ner-english",

        # Fast/lightweight
        "elastic/distilbert-base-cased-finetuned-conll03-english",

        # Modern encoder
        "Gladiator/microsoft-deberta-v3-large_ner_conll2003",

        # Larger BERT variant
        "dbmdz/bert-large-cased-finetuned-conll03-english",

        "nickprock/bert-finetuned-ner-ontonotes",

        "nickprock/distilbert-finetuned-ner-ontonotes"
    ]

    for model_id in models:
        out_file = out_dir / f"{safe_name(model_id)}_predictions.jsonl"

        run(
            runner,
            "--manifest_out", str(out_file),
            "--model", model_id,
            "--device", device,
        )


if __name__ == "__main__":
    main()