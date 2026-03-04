import json
from collections import defaultdict
from pathlib import Path

manifest_path = "../../data/processed/manifests/ner_gold_reference.jsonl"   # change to your file

label_counts = defaultdict(int)
examples = {}

with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        for ent in record.get("entities", []):
            label = ent["label"]
            label_counts[label] += 1

            # store one example text for inspection
            if label not in examples:
                examples[label] = ent["text"]

# sort labels for readability
sorted_labels = sorted(label_counts.items(), key=lambda x: x[0])

print("\nLabels found in dataset:\n")

label_dict = {}

for label, count in sorted_labels:
    label_dict[label] = {
        "description": "",
        "example": examples[label],
        "count": count
    }

    print(f"{label:10} | count={count:5} | example='{examples[label]}'")

# save template dictionary
output_file = "../../data/misc/ami_label_dictionary.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(label_dict, f, indent=4)

print(f"\nDictionary template written to {output_file}")