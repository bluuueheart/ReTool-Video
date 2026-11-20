"""Preserve generated answers, evidence and model-provided reasoning for tools."""
import ast
import re
from typing import Any

from .impl.visual_prompts import normalize_video_clip_qa_answer


def normalize_model_result(raw_response: Any, reasoning: str = "") -> dict:
    raw = str(raw_response or "")
    normalized = normalize_video_clip_qa_answer(raw)
    parsed = normalized.get("parsed_json")
    if parsed is None:
        # Some local models emit a Python-style quoted mapping plus trailing prose.
        start = normalized["cleaned_text"].find("{")
        if start >= 0:
            for end in re.finditer(r"}", normalized["cleaned_text"][start:]):
                try:
                    candidate = ast.literal_eval(normalized["cleaned_text"][start:start+end.end()])
                    if isinstance(candidate, dict):
                        parsed = candidate
                        break
                except (ValueError, SyntaxError):
                    continue
    blocks = re.findall(r"<think\b[^>]*>(.*?)</think>", raw, flags=re.I | re.S)
    result = dict(parsed) if isinstance(parsed, dict) else {}
    for key, value in list(result.items()):
        canonical = str(key).casefold()
        if canonical in {"answer", "reasoning", "explanation", "evidence", "uncertainty", "verdict"}:
            result.setdefault(canonical, value)
    supplied_reasoning = result.get("reasoning") or result.get("explanation") or ""
    model_reasoning = str(reasoning or "\n".join(blocks) or supplied_reasoning)
    answer = result.get("answer") or result.get("result") or normalized.get("cleaned_text") or raw
    result.update(raw_response=raw, reasoning=model_reasoning, parsed_json=parsed,
                  result=str(answer), reasoning_available=bool(model_reasoning))
    return result


def generate_model_result(provider, messages, **kwargs) -> dict:
    # vLLM returns reasoning in a separate response field, on the calling thread.
    from core.llm.response_metadata import clear_response_reasoning, take_response_reasoning

    clear_response_reasoning()
    from core.llm.vllm_provider import pop_last_reasoning
    pop_last_reasoning()
    response = provider.generate(messages, **kwargs)
    reasoning = take_response_reasoning() or pop_last_reasoning()
    return normalize_model_result(response, reasoning=reasoning)
