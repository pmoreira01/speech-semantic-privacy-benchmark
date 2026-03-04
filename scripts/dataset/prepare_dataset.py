import subprocess
import sys


def run(script, *args):
    cmd = [sys.executable, script] + list(args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():

    # 1 Build canonical truth dataset from raw AMI annotations
    run("scripts/dataset/build_canonical_truth.py",
        "--ne")

    # 2 Make NER manifest
    run("scripts/dataset/make_ner_manifest.py")

    # 3 Make ASR manifest
    run("scripts/dataset/make_asr_manifest.py")

    # 4 Map AMI-specific NER labels to standard ones
    run("scripts/dataset/apply_mapping_to_manifest.py")

if __name__ == "__main__":
    main()