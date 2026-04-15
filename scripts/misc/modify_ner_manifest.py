# =========================
# DEFINE CORRECTIONS
# Fill these in after reviewing the list above
# =========================

# Entities to DELETE entirely from the manifest
# Format: (segment_id, entity_text_norm, std_label)
TO_DELETE = [
    ("ES2004d_A_0553360_0556210_0041", "market",          "PER"),
    ("ES2008b_A_2117380_2137450_0146", "market",          "PER"),
    ("ES2009a_B_0731600_0734460_0029", "thirty second",   "LOC"),
    ("ES2011c_D_0335610_0355780_0007", "the",             "LOC"),
    ("ES2011c_D_0335610_0355780_0007", "of",              "LOC"),
    ("ES2011c_D_0421970_0442140_0012", "innovative so",   "PER"),
    ("ES2011c_D_0451330_0462410_0014", "good decision",   "PER"),
    ("IS1004a_A_0720950_0741110_0026", "market person",   "PER"),
    ("IS1005a_B_0099560_0102410_0002", "technic",         "PER"),
    ("IS1005b_B_1304930_1310030_0044", "management man",  "PER"),
    ("IS1008d_A_0050790_0059870_0003", "p_m_s",           "PER"),
    ("IS1003b_C_1491230_1510230_0119", "reaction",        "ORG"),
]

# Entities to RELABEL in the manifest
# Format: (segment_id, entity_text_norm, old_label, new_label)
TO_RELABEL = [
    ("ES2009a_D_0507730_0518330_0023", "chris bathgate", "ORG", "PER"),
]


def normalize_text(s):
    import pandas as pd
    import re
    if pd.isna(s) or s is None:
        return None
    return re.sub(r"\s+", " ", str(s).lower().strip())

# =========================
# APPLY AND SAVE
# =========================
import json
from pathlib import Path

MANIFEST_PATH_ORIG  = Path("data/processed/manifests/ner_manifest_mapped.jsonl")
MANIFEST_PATH_FIXED = Path("data/processed/manifests/ner_manifest_mapped_fixed.jsonl")

# Index manifest by segment_id
manifest_index = {}
with open(MANIFEST_PATH_ORIG) as f:
    for line in f:
        line = line.strip()
        if line:
            row = json.loads(line)
            manifest_index[row["segment_id"]] = row

# Build lookup sets for fast matching
delete_set  = {(s, e, l) for s, e, l in TO_DELETE}
relabel_map = {(s, e, ol): nl for s, e, ol, nl in TO_RELABEL}

n_deleted  = 0
n_relabelled = 0

for seg_id, seg in manifest_index.items():
    new_entities = []
    for ent in seg.get("entities", []):
        ent_norm  = normalize_text(ent.get("text", ""))
        std_label = ent.get("std_label", "")
        key       = (seg_id, ent_norm, std_label)

        if key in delete_set:
            n_deleted += 1
            continue  # drop it

        new_label = relabel_map.get(key)
        if new_label:
            ent["std_label"] = new_label
            n_relabelled += 1

        new_entities.append(ent)
    seg["entities"] = new_entities

# Write fixed manifest
with open(MANIFEST_PATH_FIXED, "w", encoding="utf-8") as f:
    for seg in manifest_index.values():
        f.write(json.dumps(seg, ensure_ascii=False) + "\n")

print(f"Deleted   : {n_deleted}")
print(f"Relabelled: {n_relabelled}")
print(f"Saved to  : {MANIFEST_PATH_FIXED}")