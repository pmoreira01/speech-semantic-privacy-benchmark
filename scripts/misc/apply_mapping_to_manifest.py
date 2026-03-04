import json
from pathlib import Path

MANIFEST_IN = "../../data/processed/manifests/ner_gold_reference.jsonl"
MANIFEST_OUT = "../../data/processed/manifests/ner_gold_reference_mapped.jsonl"
MAP_FILE = "ami_to_std.json"

# Policy:
# - "keep": keep all entities but add std_label (unknown -> "MISC")
# - "drop_nonstd": drop entities whose std_label is "MISC"
POLICY = "keep"

label_map = json.loads(Path(MAP_FILE).read_text(encoding="utf-8"))

def map_label(ami_label: str) -> str:
    return label_map.get(ami_label, "MISC")  # default if unseen

with open(MANIFEST_IN, "r", encoding="utf-8") as fin, open(MANIFEST_OUT, "w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        ents = rec.get("entities", [])

        mapped_ents = []
        for e in ents:
            std = map_label(e["label"])
            if POLICY == "drop_nonstd" and std == "MISC":
                continue

            e2 = dict(e)
            e2["std_label"] = std
            mapped_ents.append(e2)

        rec["entities"] = mapped_ents
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Saved mapped manifest to: {MANIFEST_OUT}")