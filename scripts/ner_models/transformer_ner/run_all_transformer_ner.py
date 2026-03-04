import subprocess
import sys


def run(script, *args):
    cmd = [sys.executable, script] + list(args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    # 1 BERT NER (baseline)
    run("scripts/ner_models/transformer_ner/run_hf_ner.py",
        "--manifest_out",
        "data/processed/ner_predictions/transformer_ner/bert_ner_predictions.jsonl",
        "--model",
        "dslim/bert-base-NER",
        "--device",
        "0"
    )

if __name__ == "__main__":
    main()