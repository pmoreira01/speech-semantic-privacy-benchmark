"""Apply the AMI-to-standard label mapping to entities in a NER manifest."""

import argparse
import json
from pathlib import Path


def map_label(label_map: dict, ami_label: str) -> str:
    return label_map.get(ami_label, "MISC")


def main():
    ap = argparse.ArgumentParser(description="Map AMI NE labels to standard NER labels.")
    ap.add_argument(
        "--manifest_in",
        default="data/processed/manifests/ner_manifest_sample1k.jsonl",
        help="Input NER manifest (JSONL)",
    )
    ap.add_argument(
        "--manifest_out",
        default="data/processed/manifests/ner_manifest_sample1k_mapped.jsonl",
        help="Output manifest with std_label added to each entity (JSONL)",
    )
    ap.add_argument(
        "--map_file",
        default="scripts/dataset/ami_to_std.json",
        help="JSON file mapping AMI label IDs to standard NER labels",
    )
    ap.add_argument(
        "--policy",
        choices=["keep", "drop_nonstd"],
        default="keep",
        help="keep: retain all entities (unknown -> MISC); drop_nonstd: discard MISC entities",
    )
    args = ap.parse_args()

    label_map = json.loads(Path(args.map_file).read_text(encoding="utf-8"))
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.manifest_in, "r", encoding="utf-8") as fin, \
         manifest_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            mapped_ents = []
            for e in rec.get("entities", []):
                std = map_label(label_map, e["label"])
                if args.policy == "drop_nonstd" and std == "MISC":
                    continue
                e2 = dict(e)
                e2["std_label"] = std
                mapped_ents.append(e2)
            rec["entities"] = mapped_ents
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved mapped manifest to: {args.manifest_out}")


if __name__ == "__main__":
    main()
