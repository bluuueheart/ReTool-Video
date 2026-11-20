"""Resolve evidence references against runtime observations, never planner history."""
import json
import re


def extract_answer_options(question):
    matches = re.findall(r"^\s*(?:\(([A-Z])\)|([A-Z])[.):])\s+(.+?)\s*$", str(question), flags=re.MULTILINE)
    return {left or right: text for left, right, text in matches}


def _references(value):
    if isinstance(value, list):
        for item in value:
            yield from _references(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"image", "image_path", "source_images", "evidence_id", "segment_ref", "resolved_segment_ref"}:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, list):
                    yield from (x for x in item if isinstance(x, str))
            elif isinstance(item, (list, dict)):
                yield from _references(item)


def validate_evidence(items, observations):
    index = {}
    for key, observation in observations.items():
        if observation.get("status") != "success":
            continue
        for reference in {key, *_references(observation.get("result"))}:
            index.setdefault(reference, []).append(key)
    checked = []
    for item in items:
        row = dict(item)
        reference = str(item.get("citation") or "").strip()
        matches = index.get(reference, [])
        quote = str(item.get("quote") or "").strip()
        if matches and quote:
            matches = [key for key in matches if quote in json.dumps(observations[key]["result"], ensure_ascii=False)]
        row.update(citation_status="resolved" if matches else "unverified", observation_ids=matches)
        if not matches and row.get("evidence_status") == "exact":
            row["evidence_status"] = "approximate"
        checked.append(row)
    return checked


def normalize_option_assessments(options, assessments, observations):
    if isinstance(options, list):
        options = {chr(65+i): value for i, value in enumerate(options)}
    options = options if isinstance(options, dict) else {}
    by_option = {str(row.get("option")): row for row in assessments or [] if isinstance(row, dict)}
    result = []
    for key, text in options.items():
        row = by_option.get(str(key), {})
        refs = row.get("citations") or []
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            refs = []
        checked = validate_evidence([{"citation": ref} for ref in refs], observations)
        valid = bool(checked) and all(item["citation_status"] == "resolved" for item in checked)
        verdict = str(row.get("verdict", "unknown")).lower()
        if verdict not in {"supported", "refuted", "unknown"} or not valid:
            verdict = "unknown"
        result.append({"option": str(key), "text": text, "verdict": verdict,
                       "citations": refs, "citations_verified": valid, "reasoning": str(row.get("reasoning") or "")})
    return result


def normalize_conflicts(conflicts, observations):
    result = []
    for row in conflicts if isinstance(conflicts, list) else []:
        row = dict(row) if isinstance(row, dict) else {"issue": str(row)}
        refs = row.get("citations")
        refs = refs if isinstance(refs, list) else []
        checked = validate_evidence([{"citation": ref} for ref in refs], observations)
        valid = len(set(str(ref) for ref in refs)) >= 2 and all(item["citation_status"] == "resolved" for item in checked)
        row["citations_verified"] = valid
        row["resolved"] = row.get("resolved") is True and valid
        result.append(row)
    return result


def observation_model_view(observations, evidence_items, max_chars=24000):
    """Keep the baseline and cited sources visible when long raw reasoning is present."""
    selected = {key for item in evidence_items for key in item.get("observation_ids", [])}
    if "initial:whole_video_qa" in observations:
        selected.add("initial:whole_video_qa")
    selected.update(list(observations)[-4:])
    keys = [key for key in observations if key in selected]
    per_source = max(512, max_chars // max(1, len(keys)))
    rows = []
    for key in keys:
        row = observations[key]
        raw = row["result"]
        payload = ({name: raw[name] for name in
                    ("answer", "verdict", "extracted_text", "result", "evidence", "sample_times", "items", "tracks", "image_evidence", "reasoning", "raw_response")
                    if name in raw} if isinstance(raw, dict) else raw)
        text = json.dumps(payload, ensure_ascii=False)
        rows.append({"evidence_id": key, "tool": row["tool"], "status": row["status"], "args": row["args"],
                     "output_excerpt": text[:per_source], "output_truncated": len(text) > per_source})
    return {"observations": rows, "omitted_observation_ids": [key for key in observations if key not in selected],
            "note": "Excerpts are model input only; full original observations remain in runtime history. Omission is not negative evidence."}
