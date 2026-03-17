import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm


DEFAULT_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_CHAT_URL = "http://localhost:11434/api/chat"

LABEL_SET = "PER|ORG|LOC|DATE|TIME|MONEY|PERCENT|CARDINAL|MISC"

PROMPT_TEMPLATE = """Return ONLY valid JSON:
{{
  "entities": [
    {{
      "label": "{LABEL_SET}",
      "text": ""
    }}
  ]
}}
Rules:
- Return only entities that appear exactly in TEXT.
- Copy entity text exactly as written in TEXT.
- Use only labels from: {LABEL_SET}
- If none, return {{"entities":[]}}.
- Do not output anything except JSON.

TEXT:
{TEXT}
"""

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_json_obj(text: str) -> Optional[dict]:
    if not text:
        return None

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def validate_and_clean_entities(text: str, entities: Any) -> List[Dict[str, Any]]:
    if not isinstance(entities, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    used_spans = set()

    for e in entities:
        if not isinstance(e, dict):
            continue

        entity_text = str(e.get("text", "")).strip()
        label = str(e.get("label", "MISC")).strip()

        if not entity_text:
            continue

        # find all exact matches
        start_search = 0
        found = False
        while True:
            start = text.find(entity_text, start_search)
            if start == -1:
                break

            end = start + len(entity_text)
            span_key = (start, end, label)

            if span_key not in used_spans:
                used_spans.add(span_key)
                cleaned.append({
                    "start_char": start,
                    "end_char": end,
                    "label": label,
                    "text": entity_text
                })
                found = True
                break

            start_search = start + 1

        if not found:
            continue

    return cleaned


def ollama_generate(session: requests.Session, url: str, model: str, prompt: str,
                    temperature: float, num_ctx: Optional[int], timeout_s: int) -> str:

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    if num_ctx is not None:
        payload["options"]["num_ctx"] = int(num_ctx)

    r = session.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()

    return r.json().get("response", "")


def ollama_chat_json(session: requests.Session, url: str, model: str, prompt: str,
                     temperature: float, num_ctx: Optional[int], timeout_s: int) -> str:

    payload: Dict[str, Any] = {
        "model": model,
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    if num_ctx is not None:
        payload["options"]["num_ctx"] = int(num_ctx)

    r = session.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()

    data = r.json()

    return (data.get("message") or {}).get("content", "")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest_in", default="data/processed/manifests/ner_manifest_sample1k.jsonl")
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--model", required=True)

    ap.add_argument("--ollama_url", default=DEFAULT_GENERATE_URL)
    ap.add_argument("--use_chat_json", action="store_true")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_ctx", type=int, default=None)
    ap.add_argument("--timeout_s", type=int, default=180)

    ap.add_argument("--max_chars", type=int, default=800)

    ap.add_argument("--no_text", action="store_true")
    ap.add_argument("--keep_raw", action="store_true")

    ap.add_argument("--retries", type=int, default=1)

    args = ap.parse_args()

    if args.use_chat_json and args.ollama_url == DEFAULT_GENERATE_URL:
        args.ollama_url = DEFAULT_CHAT_URL

    records = list(iter_jsonl(args.manifest_in))

    out_rows: List[Dict[str, Any]] = []

    with requests.Session() as session:

        for rec in tqdm(records, desc=f"LLM NER: {args.model}"):

            text = (rec.get("text", "") or "")

            if args.max_chars and len(text) > args.max_chars:
                text_for_model = text[:args.max_chars]
            else:
                text_for_model = text

            prompt = PROMPT_TEMPLATE.format(TEXT=text_for_model, LABEL_SET=LABEL_SET)

            raw = ""
            latency_ms: Optional[float] = None
            last_err: Optional[str] = None

            for _ in range(args.retries + 1):

                try:

                    t0 = time.perf_counter()

                    if args.use_chat_json:
                        raw = ollama_chat_json(
                            session,
                            args.ollama_url,
                            args.model,
                            prompt,
                            args.temperature,
                            args.num_ctx,
                            args.timeout_s,
                        )
                    else:
                        raw = ollama_generate(
                            session,
                            args.ollama_url,
                            args.model,
                            prompt,
                            args.temperature,
                            args.num_ctx,
                            args.timeout_s,
                        )

                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    last_err = None
                    break

                except Exception as e:
                    last_err = str(e)

            parsed = extract_json_obj(raw) or {"entities": []}

            cleaned = validate_and_clean_entities(
                text_for_model,
                parsed.get("entities", []),
            )

            out_rec: Dict[str, Any] = {
                "meeting_id": rec.get("meeting_id"),
                "segment_id": rec.get("segment_id"),
                "speaker_id": rec.get("speaker_id"),
                "start_time": rec.get("start_time"),
                "end_time": rec.get("end_time"),
                "overlap": rec.get("overlap"),
                "model": f"ollama:{args.model}",
                "latency_ms": latency_ms,
                "predicted_entities": cleaned,
            }

            if not args.no_text:
                out_rec["text"] = text

            if args.keep_raw:
                out_rec["raw_model_output"] = raw

            if last_err is not None:
                out_rec["error"] = last_err

            out_rows.append(out_rec)

    write_jsonl(args.manifest_out, out_rows)

    print(f"\nWrote {len(out_rows)} rows to: {args.manifest_out}")


if __name__ == "__main__":
    main()