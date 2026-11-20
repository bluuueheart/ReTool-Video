"""Time parsing shared by planner routing and video tools. Numeric values are seconds."""
import math
import re
from typing import Any, Optional, Tuple


def parse_time_seconds(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    text = re.sub(r"\s*(seconds?|secs?|sec|s)\s*$", "", str(value).strip(), flags=re.I)
    try:
        parts = [float(part) for part in text.split(":")]
        if not 1 <= len(parts) <= 3 or not all(math.isfinite(p) for p in parts):
            return None
        if len(parts) > 1 and any(p < 0 or p >= 60 for p in parts[1:]):
            return None
        return sum(part * 60 ** i for i, part in enumerate(reversed(parts)))
    except (TypeError, ValueError):
        return None


def parse_time_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        parts = value
    elif isinstance(value, str):
        parts = re.split(r"\s*(?:-->|–|—|-)\s*", value.strip().strip("[]()"), maxsplit=1)
        if len(parts) != 2:
            parts = value.strip().strip("[]()").split(",")
    else:
        return None
    if len(parts) != 2:
        return None
    start, end = (parse_time_seconds(p) for p in parts)
    return (start, end) if start is not None and end is not None else None


def normalize_visual_time_args(args: dict) -> dict:
    result = dict(args)
    for source, target in (
        ("timestamp_start", "t_start"), ("timestamp_end", "t_end"),
        ("time_range_start", "t_start"), ("time_range_end", "t_end"),
        ("start_time", "t_start"), ("end_time", "t_end"),
    ):
        if result.get(target) in (None, "") and result.get(source) not in (None, ""):
            result[target] = result[source]
        result.pop(source, None)
    for key in ("time_range", "timestamp_range", "search_window"):
        window = parse_time_range(result.get(key))
        if window:
            result.setdefault("t_start", window[0])
            result.setdefault("t_end", window[1])
            result.pop(key, None)
    for key in ("t_start", "t_end", "timestamp"):
        if key not in result:
            continue
        def convert(value):
            if isinstance(value, list):
                return [convert(item) for item in value]
            parsed = parse_time_seconds(value)
            if parsed is None:
                raise ValueError(f"{key} must be seconds or HH:MM:SS, received {value!r}")
            return parsed
        if result[key] is not None:
            result[key] = convert(result[key])
    return result


def annotate_indexed_segment_time(record, *, source):
    """Attach absolute bounds at a trusted preprocessing-index producer, never infer clip time."""
    bounds = parse_time_range(record.get("time_range") or record.get("time"))
    if bounds is None or not 0 <= bounds[0] <= bounds[1]:
        return record
    return {**record, "start": bounds[0], "end": bounds[1], "timestamp_basis": "absolute",
            "clip_offset_seconds": 0.0, "source_window": {"start": bounds[0], "end": bounds[1]},
            "timestamp_source": source}


def normalize_temporal_records(records, *, require_basis=False):
    """Validate ranges and convert explicitly clip-relative records once, preserving provenance."""
    if not isinstance(records, list):
        raise ValueError("Temporal records must be an array.")
    normalized = []
    for index, original in enumerate(records):
        if not isinstance(original, dict):
            raise ValueError(f"Temporal record {index} must be an object with start/end.")
        row = dict(original)
        if row.get("basis_uncertain"):
            raise ValueError(f"Temporal record {index} has an uncertain timestamp basis.")
        basis = row.get("timestamp_basis")
        if require_basis and basis not in {"absolute", "clip_relative"}:
            raise ValueError(f"Video event record {index} needs timestamp_basis. Pass actual evidence_events records, including source_window and clip_offset_seconds, rather than retyping timestamps.")
        if require_basis and row.get("source_window") is None:
            raise ValueError(f"Video event record {index} needs its source_window; preserve the window metadata from evidence_events.")
        if basis not in {None, "absolute", "clip_relative"}:
            raise ValueError(f"Unknown timestamp_basis in temporal record {index}: {basis!r}")
        start, end = parse_time_seconds(row.get("start")), parse_time_seconds(row.get("end"))
        if start is None or end is None or end < start:
            raise ValueError(f"Temporal record {index} needs valid start/end seconds with end >= start.")
        if basis == "clip_relative":
            offset = parse_time_seconds(row.get("clip_offset_seconds"))
            if offset is None:
                raise ValueError(f"Clip-relative record {index} needs clip_offset_seconds.")
            row.setdefault("clip_start", start)
            row.setdefault("clip_end", end)
            start, end = start + offset, end + offset
            row["timestamp_basis"] = "absolute"
        window = row.get("source_window")
        if window is not None:
            bounds = parse_time_range([window.get("start"), window.get("end")]) if isinstance(window, dict) else parse_time_range(window)
            if bounds is None or not bounds[0] - 1 <= start <= end <= bounds[1] + 1:
                raise ValueError(f"Temporal record {index} falls outside its source_window.")
            row["source_window"] = {"start": bounds[0], "end": bounds[1]}
        row.update(start=start, end=end)
        normalized.append(row)
    return normalized
