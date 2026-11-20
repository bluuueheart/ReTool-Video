"""Turn clip-relative timestamps quoted in a Video_Clip_QA evidence list into absolute-second records.

Video_Clip_QA cuts [start, end] out of the video, so the model's "01:32" means 92s after the clip start, not 92s in the
full video. Prose evidence therefore cannot be merged across windows without an offset. This module adds numeric
`{start, end}` records (absolute seconds, the keys Merge_Temporal_Segments / Sort_Time_Ranges consume) next to the
untouched prose; nothing is inferred beyond the timestamps the model itself wrote.
"""
import re
from typing import Any, Dict, List, Optional

from toolkit.time_utils import parse_time_seconds

_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?"
_TOKEN = re.compile(rf"(?<![\d:.]){_TIME}(?!\d|:\d)")
_RANGE = re.compile(
    rf"(?<![\d:.])({_TIME})(?!\d|:\d)\s*(?:and\s+(?:continues?|lasts?)\s+(?:until|to)|to|and|until|through|-|–|—|-->)\s*(?<![\d:.])({_TIME})(?!\d|:\d)", re.I)

BASIS = "clip_relative"
BASIS_NOTE = ("Times written inside answer/evidence text are relative to the start of the queried window; "
              "`evidence_events` holds the same times as absolute seconds in the full video.")


def _event(text: str, rel_start: float, rel_end: float, offset: float, duration: float) -> Dict[str, Any]:
    event: Dict[str, Any] = {"text": text, "clip_start": round(rel_start, 3), "clip_end": round(rel_end, 3)}
    if max(rel_start, rel_end) <= duration + 1.0:  # Beyond the window the model clearly did not use clip time.
        event["start"] = round(offset + rel_start, 3)
        event["end"] = round(offset + rel_end, 3)
        event["timestamp_basis"] = "absolute"
        event["clip_offset_seconds"] = round(offset, 3)
        event["source_window"] = {"start": round(offset, 3), "end": round(offset + duration, 3)}
    else:
        event["basis_uncertain"] = "time is later than the queried window; not converted"
    return event


def extract_evidence_events(evidence: Any, offset: float, duration: float) -> List[Dict[str, Any]]:
    """One event per evidence line that quotes timestamps: a range if it names two ends, else its first point."""
    items = evidence if isinstance(evidence, (list, tuple)) else [evidence]
    events: List[Dict[str, Any]] = []
    for item in items:
        text = item if isinstance(item, str) else str(item or "")
        if not text.strip():
            continue
        found = _RANGE.search(text)
        if found:
            start, end = parse_time_seconds(found.group(1)), parse_time_seconds(found.group(2))
            if start is not None and end is not None:
                events.append(_event(text.strip(), min(start, end), max(start, end), offset, duration))
            continue  # A malformed range (e.g. 00:90) must not degrade into a misleading single point.
        for token in _TOKEN.findall(text):
            point = parse_time_seconds(token)
            if point is not None:
                events.append(_event(text.strip(), point, point, offset, duration))
                break
    return events


def annotate_clip_times(result: Dict[str, Any], offset: float, duration: float,
                        evidence: Optional[Any] = None) -> Dict[str, Any]:
    """Add timestamp basis metadata (always) and `evidence_events` (when the evidence quotes times)."""
    result["timestamp_basis"] = BASIS
    result["clip_offset_seconds"] = round(float(offset), 3)
    events = extract_evidence_events(evidence if evidence is not None else result.get("evidence_items"),
                                     float(offset), float(duration))
    if events:
        result["evidence_events"] = events
        result["timestamp_note"] = BASIS_NOTE
    return result
