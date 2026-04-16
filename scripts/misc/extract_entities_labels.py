import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import iter_jsonl


def main():
    ap = argparse.ArgumentParser(description="Print entity label statistics for a NER manifest.")
    ap.add_argument(
        "--manifest_in",
        default="data/processed/manifests/ner_gold_reference.jsonl",
        help="Input NER JSONL manifest",
    )
    ap.add_argument(
        "--output_file",
        default="data/misc/ami_label_dictionary.json",
        help="Path to write the label dictionary JSON",
    )
    args = ap.parse_args()

    label_counts = defaultdict(int)
    examples = {}

    for record in iter_jsonl(args.manifest_in):
        for ent in record.get("entities", []):
            label = ent["label"]
            label_counts[label] += 1
            if label not in examples:
                examples[label] = ent["text"]

    sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])

    print("\nLabels found in dataset:\n")

    label_dict = {}
    for label, count in sorted_labels:
        label_dict[label] = {
            "description": "",
            "example": examples[label],
            "count": count,
        }
        print(f"{label:10} | count={count:5} | example='{examples[label]}'")

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=4)

    print(f"\nDictionary template written to {output_file}")


if __name__ == "__main__":
    main()
