"""Per-thread optional response metadata without changing generate()'s string API."""
import threading

_response = threading.local()


def clear_response_reasoning():
    _response.reasoning = ""


def record_response_reasoning(value):
    _response.reasoning = str(value or "")


def take_response_reasoning():
    value = str(getattr(_response, "reasoning", "") or "")
    clear_response_reasoning()
    return value
