#!/usr/bin/env python3
"""
Build canonical "truth" meeting JSON for AMI (NXT) for ASR+NER benchmarking.

What it does
------------
For each meeting_id:
  1) Finds per-speaker *.words.xml transcript files and parses a timed word stream.
  2) Derives utterances using pause-based segmentation (configurable).
  3) Optionally finds named-entity annotation files, extracts NE spans by resolving word-range pointers,
     then attaches entities to utterances and computes character offsets.

Outputs
-------
data/processed/meetings/<meeting_id>.truth.json  (by default)

Assumptions / notes
-------------------
- AMI word transcripts are typically distributed as NXT-formatted *.words.xml per speaker
  (e.g., EN2001a.A.words.xml) and include forced alignment timings. See AMI docs. :contentReference[oaicite:2]{index=2}
- Named-entity files are not guaranteed present for all meetings; this script supports several common
  NXT patterns, but your local distribution may differ.
- This script treats "canonical truth" as immutable *once generated*, but it uses a reproducible
  utterance segmentation policy (pause threshold, max utterance length, etc.).

You should version-control:
- This script
- Your config choices (CLI args)
But do NOT version-control the raw AMI distribution itself.

Usage examples
--------------
1) Build for all meetings found under AMI root:
    python scripts/build_canonical_truth.py --ami-root data/raw/ami --out-dir data/processed/meetings

2) Build only some meetings:
    python scripts/build_canonical_truth.py --ami-root /path/to/ami --meetings ES2002a IS1001a

3) With named entities (if present) and ontology mapping:
    python scripts/build_canonical_truth.py --ami-root /path/to/ami --ne --ne-ontology /path/to/ontologies/ne-types.xml

"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET


NITE_NS = "http://nite.sourceforge.net/"
XML_NS = "http://www.w3.org/XML/1998/namespace"

# ElementTree namespace handling:
NS = {
    "nite": NITE_NS,
}

# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class Word:
    word_id: str
    text: str
    start: Optional[float]
    end: Optional[float]
    speaker_id: str


@dataclass
class Utterance:
    utterance_id: str
    meeting_id: str
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    tokens: List[Dict]
    entities: List[Dict]
    overlap: bool = False


@dataclass(frozen=True)
class NamedEntitySpan:
    meeting_id: str
    speaker_id: str
    source_file: str
    ne_id: str
    label: str
    word_ids: List[str]
    start_time: float
    end_time: float
    text: str


# ---------------------------
# Utilities
# ---------------------------

def safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    x = x.strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_xml(path: Path) -> ET.Element:
    # ET can be sensitive to weird encodings; let it raise with a clean message.
    try:
        tree = ET.parse(str(path))
        return tree.getroot()
    except ET.ParseError as e:
        raise RuntimeError(f"XML parse error in {path}: {e}") from e


def get_attr_any(elem: ET.Element, names: List[str]) -> Optional[str]:
    for n in names:
        if n in elem.attrib:
            return elem.attrib[n]
        # try namespaced
        for k, v in elem.attrib.items():
            if k.endswith("}" + n):  # crude fallback
                return v
    return None


def get_nite_id(elem: ET.Element) -> Optional[str]:
    # NXT commonly uses nite:id attribute
    # In ElementTree, namespaced attributes look like '{namespace}id'
    for k, v in elem.attrib.items():
        if k.endswith("}id") or k == "nite:id":
            return v
    return None


def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s


# ---------------------------
# Locate AMI files
# ---------------------------

WORDS_RE = re.compile(r"^(?P<meeting>[A-Z]{2}\d{4}[a-z])\.(?P<speaker>[A-Z])\.words\.xml$", re.IGNORECASE)

def find_words_files(ami_root: Path) -> List[Path]:
    return sorted(ami_root.rglob("*.words.xml"))


def infer_meeting_and_speaker_from_words_filename(p: Path) -> Optional[Tuple[str, str]]:
    m = WORDS_RE.match(p.name)
    if not m:
        return None
    return (m.group("meeting"), m.group("speaker"))


def group_words_files_by_meeting(ami_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Returns:
      { meeting_id: { speaker_id: path_to_words_xml } }
    """
    out: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for f in find_words_files(ami_root):
        inf = infer_meeting_and_speaker_from_words_filename(f)
        if not inf:
            continue
        meeting_id, speaker_id = inf
        out[meeting_id][speaker_id] = f
    return dict(out)


# ---------------------------
# Parse words.xml
# ---------------------------

def parse_words_file(words_path: Path, meeting_id: str, speaker_id: str) -> List[Word]:
    """
    Parse NXT words file.

    Common word elements include <w ...>text</w>
    Attributes often include starttime/endtime.

    We keep only elements that look like words (tag endswith 'w') and have non-empty text.
    """
    root = read_xml(words_path)

    words: List[Word] = []

    # In NXT, w elements may be nested; get all.
    for elem in root.iter():
        tag = elem.tag
        # handle namespaces: '{ns}w' or 'w'
        local = tag.split("}")[-1]
        if local != "w":
            continue

        wid = get_nite_id(elem) or elem.get("id") or elem.get(f"{{{XML_NS}}}id")
        if not wid:
            # fall back to a deterministic synthetic id (rare)
            wid = f"{meeting_id}.{speaker_id}.w.synthetic.{len(words)}"

        txt = (elem.text or "").strip()
        if not txt:
            # keep silent/nonword tokens out; you can change if you want
            continue

        start = safe_float(elem.get("starttime") or elem.get("start") or elem.get("stime"))
        end = safe_float(elem.get("endtime") or elem.get("end") or elem.get("etime"))

        words.append(
            Word(
                word_id=wid,
                text=txt,
                start=start,
                end=end,
                speaker_id=speaker_id,
            )
        )

    # Sort by start time where possible; otherwise stable by file order.
    # Many words have both times; some may have one missing.
    def sort_key(w: Word):
        st = w.start if w.start is not None else float("inf")
        en = w.end if w.end is not None else float("inf")
        return (st, en)

    words_sorted = sorted(words, key=sort_key)
    return words_sorted


# ---------------------------
# Utterance segmentation
# ---------------------------

def segment_into_utterances(
    meeting_id: str,
    speaker_id: str,
    words: List[Word],
    pause_threshold_s: float = 0.7,
    max_utt_s: float = 20.0,
    max_words: int = 80,
) -> List[Utterance]:
    """
    Deterministic utterance segmentation based on pauses and max length constraints.

    - Start a new utterance if gap between consecutive words > pause_threshold_s
    - Or if utterance would exceed max_utt_s
    - Or if utterance would exceed max_words

    Requires word times; if missing, falls back to word-count-only segmentation.
    """
    utterances: List[Utterance] = []
    cur: List[Word] = []

    def flush():
        nonlocal cur
        if not cur:
            return
        # Compute times (best-effort).
        starts = [w.start for w in cur if w.start is not None]
        ends = [w.end for w in cur if w.end is not None]
        if not starts or not ends:
            # Fallback: synthetic times (0), but keep order.
            st = 0.0
            en = 0.0
        else:
            st = float(min(starts))
            en = float(max(ends))

        text = " ".join(w.text for w in cur).strip()
        # Create token list; if timing missing, set None.
        tokens = [{"text": w.text, "start": w.start, "end": w.end, "word_id": w.word_id} for w in cur]

        utt_id = make_utterance_id(meeting_id, speaker_id, st, en, len(utterances))
        utterances.append(
            Utterance(
                utterance_id=utt_id,
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                start_time=st,
                end_time=en,
                text=text,
                tokens=tokens,
                entities=[],
                overlap=False,
            )
        )
        cur = []

    for w in words:
        if not cur:
            cur.append(w)
            continue

        prev = cur[-1]
        gap = None
        if prev.end is not None and w.start is not None:
            gap = w.start - prev.end

        # determine current utterance duration if we have times
        cur_start = next((x.start for x in cur if x.start is not None), None)
        cur_end = next((x.end for x in reversed(cur) if x.end is not None), None)
        cur_dur = (cur_end - cur_start) if (cur_start is not None and cur_end is not None) else None

        must_split = False
        if gap is not None and gap > pause_threshold_s:
            must_split = True
        if cur_dur is not None and (cur_dur > max_utt_s):
            must_split = True
        if len(cur) >= max_words:
            must_split = True

        if must_split:
            flush()
            cur.append(w)
        else:
            cur.append(w)

    flush()
    return utterances


def make_utterance_id(meeting_id: str, speaker_id: str, start_s: float, end_s: float, idx: int) -> str:
    # Use milliseconds to avoid float drift; include idx for uniqueness when times are missing.
    start_ms = int(round(start_s * 1000.0))
    end_ms = int(round(end_s * 1000.0))
    return f"{meeting_id}_{speaker_id}_{start_ms:07d}_{end_ms:07d}_{idx:04d}"


# ---------------------------
# Named entity parsing (optional)
# ---------------------------

# Common NXT href range patterns:
# 1) "ES2002a.B.words.xml#id(ES2002a.B.words4)..id(ES2002a.B.words16)"
# 2) "o1.words.xml#xpointer(id('w_1')/range-to(id('w_5')))"
HREF_RANGE_RE = re.compile(r"#id\((?P<start>[^)]+)\)\.\.id\((?P<end>[^)]+)\)")
XPTR_RANGE_RE = re.compile(r"#xpointer\(id\('(?P<start>[^']+)'\)/range-to\(id\('(?P<end>[^']+)'\)\)\)")
HREF_SINGLE_RE = re.compile(r"#id\((?P<single>[^)]+)\)$")
XPTR_SINGLE_RE = re.compile(r"#xpointer\(id\('(?P<single>[^']+)'\)\)$")


def parse_href_range(href: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (target_filename, start_word_id, end_word_id).
    If href points to a single id, start=end.
    """
    # href might be relative file + fragment
    if "#" in href:
        target_file, frag = href.split("#", 1)
        frag = "#" + frag
    else:
        target_file, frag = href, ""

    m = HREF_RANGE_RE.search(frag)
    if m:
        return (target_file, m.group("start"), m.group("end"))
    m = XPTR_RANGE_RE.search(frag)
    if m:
        return (target_file, m.group("start"), m.group("end"))
    m = HREF_SINGLE_RE.search(frag)
    if m:
        sid = m.group("single")
        return (target_file, sid, sid)
    m = XPTR_SINGLE_RE.search(frag)
    if m:
        sid = m.group("single")
        return (target_file, sid, sid)

    return (target_file, None, None)


def build_word_index(all_words_by_speaker: Dict[str, List[Word]]) -> Dict[str, Word]:
    idx: Dict[str, Word] = {}
    for speaker, ws in all_words_by_speaker.items():
        for w in ws:
            idx[w.word_id] = w
    return idx


def resolve_word_range(word_ids_in_order: List[str], start_id: str, end_id: str) -> List[str]:
    """
    Resolve an inclusive range start..end based on word order list.
    If ids not found, return [].
    """
    try:
        i = word_ids_in_order.index(start_id)
        j = word_ids_in_order.index(end_id)
    except ValueError:
        return []
    if i <= j:
        return word_ids_in_order[i : j + 1]
    else:
        # sometimes ranges might be reversed; handle gracefully
        return word_ids_in_order[j : i + 1]


def load_ne_ontology(ne_ontology_path: Optional[Path]) -> Dict[str, str]:
    """
    Optional: map ne-type ids (e.g., 'ne_11') -> human-readable name if present.
    If we can't find a name, we use the id.
    """
    if not ne_ontology_path:
        return {}
    if not ne_ontology_path.exists():
        return {}

    root = read_xml(ne_ontology_path)
    mapping: Dict[str, str] = {}

    for elem in root.iter():
        nid = get_nite_id(elem)
        if not nid:
            continue
        # often attribute @name exists; otherwise use element text
        name = elem.get("name") or (elem.text or "").strip()
        if name:
            mapping[nid] = name
        else:
            mapping[nid] = nid
    return mapping


def find_named_entity_files(ami_root: Path, meeting_id: str) -> List[Path]:
    """
    Heuristic search for NE files. Different distributions name these differently.
    We look for files containing meeting_id and common ne strings.
    """
    candidates: List[Path] = []
    patterns = [
        f"**/{meeting_id}.*named*entity*.xml",
        f"**/{meeting_id}.*named-entity*.xml",
        f"**/{meeting_id}.*ne*.xml",
        f"**/{meeting_id}.*namedEntities*.xml",
    ]
    for pat in patterns:
        candidates.extend(ami_root.glob(pat))
    # De-dup
    uniq = sorted({c for c in candidates if c.is_file()})
    return uniq


def extract_named_entities(
    ne_files: List[Path],
    meeting_id: str,
    word_index: Dict[str, Word],
    word_order_by_speaker: Dict[str, List[str]],
    ne_ontology: Dict[str, str],
) -> List[NamedEntitySpan]:
    """
    Parse NXT-style named-entity annotations.

    We try to find elements named 'named-entity' and extract:
    - an id (nite:id)
    - ne-type (via nite:pointer role='ne-type' or attribute)
    - covered words (via nite:child href range -> word ids)
    """
    out: List[NamedEntitySpan] = []

    for f in ne_files:
        root = read_xml(f)

        for elem in root.iter():
            local = elem.tag.split("}")[-1]
            if local not in ("named-entity", "named_entity", "namedentity", "ne", "entity"):
                continue

            ne_id = get_nite_id(elem) or f"{meeting_id}.ne.synthetic.{len(out)}"

            # label resolution:
            label_id = None
            label_text = None

            # (a) nite:pointer role="ne-type"
            for ptr in elem.findall(".//nite:pointer", NS):
                role = ptr.get("role") or ""
                href = ptr.get("href") or ptr.get(f"{{{NITE_NS}}}href")
                if "ne-type" in role or "netype" in role or "type" == role:
                    if href and "#id(" in href:
                        # ...ne-types.xml#id(ne_11)
                        m = re.search(r"#id\(([^)]+)\)", href)
                        if m:
                            label_id = m.group(1)
                    elif href:
                        # ...#something
                        frag = href.split("#")[-1]
                        label_id = frag
                    break

            # (b) attribute fallback
            if label_id is None:
                label_id = elem.get("type") or elem.get("label") or elem.get("ne-type") or elem.get("netype")

            if label_id:
                label_text = ne_ontology.get(label_id, label_id)
            else:
                label_text = "UNKNOWN"

            # coverage: find nite:child href
            hrefs = []
            for ch in elem.findall(".//nite:child", NS):
                href = ch.get("href") or ch.get(f"{{{NITE_NS}}}href")
                if href:
                    hrefs.append(href)

            # Some NXT corpora use xlink:href; check for any attribute ending with 'href'
            if not hrefs:
                for ch in elem.iter():
                    for k, v in ch.attrib.items():
                        if k.endswith("href") and isinstance(v, str):
                            hrefs.append(v)

            covered_word_ids: List[str] = []
            for href in hrefs:
                target_file, start_wid, end_wid = parse_href_range(href)
                if not start_wid or not end_wid:
                    continue

                # Determine speaker from start_wid if possible; else from target_file name
                # We'll attempt to detect speaker by matching known word IDs.
                spk = None
                if start_wid in word_index:
                    spk = word_index[start_wid].speaker_id
                else:
                    # filename often contains ".A.words.xml"
                    mm = re.search(r"\.([A-Z])\.words\.xml", target_file)
                    if mm:
                        spk = mm.group(1)

                if not spk or spk not in word_order_by_speaker:
                    continue

                rng = resolve_word_range(word_order_by_speaker[spk], start_wid, end_wid)
                if rng:
                    covered_word_ids.extend(rng)

            # De-dup preserving order
            seen = set()
            cov = []
            for wid in covered_word_ids:
                if wid not in seen:
                    cov.append(wid)
                    seen.add(wid)

            if not cov:
                continue

            # Compute times/text
            words = [word_index[wid] for wid in cov if wid in word_index]
            if not words:
                continue

            starts = [w.start for w in words if w.start is not None]
            ends = [w.end for w in words if w.end is not None]
            if not starts or not ends:
                continue

            start_time = float(min(starts))
            end_time = float(max(ends))
            speaker_id = words[0].speaker_id
            text = " ".join(w.text for w in words).strip()

            out.append(
                NamedEntitySpan(
                    meeting_id=meeting_id,
                    speaker_id=speaker_id,
                    source_file=str(f),
                    ne_id=ne_id,
                    label=label_text,
                    word_ids=cov,
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                )
            )

    return out


# ---------------------------
# Attach entities to utterances & compute char offsets
# ---------------------------

def attach_entities_to_utterances(
    utterances: List[Utterance],
    entities: List[NamedEntitySpan],
) -> None:
    """
    For each entity, find the utterance with same speaker whose time span overlaps most,
    then compute char offsets by substring search on utterance.text.

    Note: If the same entity text appears multiple times in the utterance, we choose the first
    match; for more robust alignment, you can map via word_ids -> token positions -> char offsets.
    """
    # index utterances by speaker
    utt_by_speaker: Dict[str, List[Utterance]] = defaultdict(list)
    for u in utterances:
        utt_by_speaker[u.speaker_id].append(u)

    for spk in utt_by_speaker:
        utt_by_speaker[spk].sort(key=lambda u: (u.start_time, u.end_time))

    for ent in entities:
        candidates = utt_by_speaker.get(ent.speaker_id, [])
        if not candidates:
            continue

        # choose by max temporal overlap
        best = None
        best_ov = 0.0
        for u in candidates:
            ov = max(0.0, min(u.end_time, ent.end_time) - max(u.start_time, ent.start_time))
            if ov > best_ov:
                best_ov = ov
                best = u

        if best is None or best_ov <= 0.0:
            continue

        # compute char offsets
        utt_text = best.text
        ent_text = ent.text

        start_char = utt_text.find(ent_text)
        if start_char < 0:
            # fallback: case-insensitive match
            start_char = utt_text.lower().find(ent_text.lower())
        if start_char < 0:
            # If still not found, skip attaching but keep for diagnostics in logs.
            continue

        end_char = start_char + len(ent_text)

        best.entities.append(
            {
                "entity_id": ent.ne_id,
                "start_char": start_char,
                "end_char": end_char,
                "label": ent.label,
                "text": utt_text[start_char:end_char],
                "source": ent.source_file,
            }
        )


# ---------------------------
# Overlap computation (optional, simple)
# ---------------------------

def compute_overlap_flags(all_utterances: List[Utterance]) -> None:
    """
    Mark utterances as overlap=True if any utterance from a different speaker overlaps in time.
    """
    # sort across meeting
    all_utterances.sort(key=lambda u: (u.start_time, u.end_time))
    for i, u in enumerate(all_utterances):
        u.overlap = False

    # sweep line naive O(n^2) within local window
    j = 0
    for i in range(len(all_utterances)):
        u = all_utterances[i]
        # move j to first that could overlap by start time
        j = max(j, i + 1)
        while j < len(all_utterances) and all_utterances[j].start_time <= u.end_time:
            v = all_utterances[j]
            if v.speaker_id != u.speaker_id:
                if min(u.end_time, v.end_time) - max(u.start_time, v.start_time) > 0:
                    u.overlap = True
                    v.overlap = True
            j += 1
        # reset j to i+1 for next loop (keeps window small-ish but still safe)
        j = i + 1


# ---------------------------
# Main build
# ---------------------------

def build_meeting_truth(
    ami_root: Path,
    out_dir: Path,
    meeting_id: str,
    words_files_by_speaker: Dict[str, Path],
    do_ne: bool,
    ne_ontology_path: Optional[Path],
    pause_threshold_s: float,
    max_utt_s: float,
    max_words: int,
) -> Dict:
    # Parse all speakers
    all_words_by_speaker: Dict[str, List[Word]] = {}
    for spk, path in sorted(words_files_by_speaker.items()):
        all_words_by_speaker[spk] = parse_words_file(path, meeting_id, spk)

    # Build utterances per speaker and merge
    utterances: List[Utterance] = []
    for spk, ws in all_words_by_speaker.items():
        utterances.extend(
            segment_into_utterances(
                meeting_id=meeting_id,
                speaker_id=spk,
                words=ws,
                pause_threshold_s=pause_threshold_s,
                max_utt_s=max_utt_s,
                max_words=max_words,
            )
        )

    # Compute overlap flag (simple)
    compute_overlap_flags(utterances)

    # Named entities (optional)
    if do_ne:
        ne_ontology = load_ne_ontology(ne_ontology_path)
        word_index = build_word_index(all_words_by_speaker)
        word_order_by_speaker = {
            spk: [w.word_id for w in ws] for spk, ws in all_words_by_speaker.items()
        }
        ne_files = find_named_entity_files(ami_root, meeting_id)
        entities = extract_named_entities(
            ne_files=ne_files,
            meeting_id=meeting_id,
            word_index=word_index,
            word_order_by_speaker=word_order_by_speaker,
            ne_ontology=ne_ontology,
        )
        attach_entities_to_utterances(utterances, entities)

    # Duration (best effort)
    if utterances:
        duration = float(max(u.end_time for u in utterances))
    else:
        duration = 0.0

    # Assemble output JSON
    meeting = {
        "meeting_id": meeting_id,
        "duration_seconds": duration,
        "speakers": [{"speaker_id": s, "role": "participant"} for s in sorted(words_files_by_speaker.keys())],
        "utterances": [
            {
                "utterance_id": u.utterance_id,
                "speaker_id": u.speaker_id,
                "start_time": u.start_time,
                "end_time": u.end_time,
                "text": u.text,
                "tokens": [
                    {"text": t["text"], "start": t["start"], "end": t["end"], "word_id": t["word_id"]}
                    for t in u.tokens
                ],
                "entities": u.entities,
                "overlap": u.overlap,
            }
            for u in sorted(utterances, key=lambda x: (x.start_time, x.end_time, x.speaker_id))
        ],
        "provenance": {
            "ami_root": str(ami_root.resolve()),
            "words_files": {spk: str(p) for spk, p in words_files_by_speaker.items()},
            "utterance_segmentation": {
                "pause_threshold_s": pause_threshold_s,
                "max_utt_s": max_utt_s,
                "max_words": max_words,
            },
            "named_entities": {
                "enabled": bool(do_ne),
                "ontology": str(ne_ontology_path) if ne_ontology_path else None,
            },
        },
    }

    # Write
    ensure_dir(out_dir)
    out_path = out_dir / f"{meeting_id}.truth.json"
    out_path.write_text(json.dumps(meeting, ensure_ascii=False, indent=2), encoding="utf-8")
    return meeting


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami-root", type=Path, default=Path("data/raw/amicorpus"), help="Root of AMI distribution (recursive search).")
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/meetings"), help="Output directory for truth JSON.")
    ap.add_argument("--meetings", nargs="*", default=None, help="Meeting IDs to process; if omitted, process all found in words.xml.")
    ap.add_argument("--ne", action="store_true", help="Try to attach named entities if NE files exist.")
    ap.add_argument("--ne-ontology", type=Path, default=None, help="Optional path to ne-types.xml to map type ids to names.")
    ap.add_argument("--pause-threshold", type=float, default=0.7, help="Pause (s) that triggers a new utterance.")
    ap.add_argument("--max-utt-s", type=float, default=20.0, help="Max utterance duration (s) before splitting.")
    ap.add_argument("--max-words", type=int, default=80, help="Max words per utterance before splitting.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ami_root: Path = args.ami_root
    out_dir: Path = args.out_dir

    if not ami_root.exists():
        print(f"ERROR: --ami-root does not exist: {ami_root}", file=sys.stderr)
        sys.exit(2)

    meetings_map = group_words_files_by_meeting(ami_root)

    if not meetings_map:
        print(
            "ERROR: No *.words.xml files found. Check your AMI root path and distribution.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.meetings:
        target_meetings = [m for m in args.meetings if m in meetings_map]
        missing = [m for m in args.meetings if m not in meetings_map]
        if missing:
            print(f"WARNING: meetings not found (no words.xml): {missing}", file=sys.stderr)
    else:
        target_meetings = sorted(meetings_map.keys())

    total = 0
    for meeting_id in target_meetings:
        words_files_by_speaker = meetings_map[meeting_id]
        if not words_files_by_speaker:
            continue
        build_meeting_truth(
            ami_root=ami_root,
            out_dir=out_dir,
            meeting_id=meeting_id,
            words_files_by_speaker=words_files_by_speaker,
            do_ne=bool(args.ne),
            ne_ontology_path=args.ne_ontology,
            pause_threshold_s=float(args.pause_threshold),
            max_utt_s=float(args.max_utt_s),
            max_words=int(args.max_words),
        )
        total += 1
        print(f"[OK] {meeting_id} -> {out_dir}/{meeting_id}.truth.json")

    print(f"Done. Wrote {total} meeting truth files to {out_dir}")


if __name__ == "__main__":
    main()