"""Shared utilities for the NER benchmark pipeline."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Iterate over records in a JSONL file, skipping blank lines."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e


def write_jsonl(path: Union[str, Path], rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts to a JSONL file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_name(name: str) -> str:
    """Convert a model ID or path into a filesystem-safe string."""
    result = name.replace("/", "_").replace(":", "_").replace("+", "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result


def run_subprocess(script: str, *args: str) -> None:
    """Run a Python script as a subprocess with the current interpreter."""
    cmd = [sys.executable, script] + list(args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
