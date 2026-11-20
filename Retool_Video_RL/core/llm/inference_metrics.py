"""Optional request-level accounting; no prompts, sampling settings or outputs are changed."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import fcntl
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

_scope = ContextVar('inference_metrics_scope', default={})
_lock = threading.Lock()
logger = logging.getLogger(__name__)


def configure_sample(logs_dir, sample_uid):
    if os.environ.get('TOOLSTAR_INFERENCE_METRICS') != '1':
        return
    os.environ['TOOLSTAR_INFERENCE_METRICS_PATH'] = str(Path(logs_dir) / 'inference_metrics.jsonl')
    os.environ['TOOLSTAR_METRICS_SAMPLE_UID'] = sample_uid
    os.environ['TOOLSTAR_METRICS_SAMPLE_ATTEMPT'] = uuid.uuid4().hex
    emit(kind='sample_start')


@contextmanager
def inference_scope(**values):
    token = _scope.set({**_scope.get(), **values})
    try:
        yield
    finally:
        _scope.reset(token)


def current_scope():
    return dict(_scope.get())


def emit(**values):
    path = os.environ.get('TOOLSTAR_INFERENCE_METRICS_PATH')
    if os.environ.get('TOOLSTAR_INFERENCE_METRICS') != '1' or not path:
        return
    try:
        record = {'timestamp_unix': time.time(), 'pid': os.getpid(),
            'thread': threading.current_thread().name,
            'sample_uid': os.environ.get('TOOLSTAR_METRICS_SAMPLE_UID'),
            'sample_attempt_id': os.environ.get('TOOLSTAR_METRICS_SAMPLE_ATTEMPT'),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), **_scope.get(), **values}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock, target.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as exc:
        logger.warning('Inference accounting failed: %s', exc)


def metered_stage(actor, **fields):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            started = time.perf_counter()
            with inference_scope(actor=actor, **fields):
                status = 'success'
                try:
                    return function(*args, **kwargs)
                except BaseException:
                    status = 'error'
                    raise
                finally:
                    emit(kind='stage_span', status=status, elapsed_seconds=time.perf_counter() - started)
        return call
    return decorate


def _record_request(*, provider_role, call_id, attempt, attempt_started, generation_started,
                   payload, endpoint, response=None, error=None, final_attempt=False):
    usage = getattr(response, 'usage', None) if response is not None else None
    if usage is not None:
        usage = usage.model_dump() if hasattr(usage, 'model_dump') else dict(usage)
    choices = getattr(response, 'choices', None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, 'message', None)
    content = getattr(message, 'content', '') or ''
    reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None) or ''
    modalities = {}
    for msg in payload.get('messages', []):
        for item in msg.get('content', []) if isinstance(msg.get('content'), list) else []:
            if isinstance(item, dict):
                key = item.get('type', 'unknown')
                modalities[key] = modalities.get(key, 0) + 1
    emit(kind='llm_request', provider_role=provider_role, generation_call_id=call_id,
        attempt=attempt, status='error' if error else 'success', endpoint=endpoint,
        model=payload.get('model'), request_seed=payload.get('seed'), max_tokens=payload.get('max_tokens'),
        temperature=payload.get('temperature'),
        thinking=(payload.get('extra_body') or {}).get('chat_template_kwargs'),
        elapsed_seconds=time.perf_counter() - attempt_started,
        generation_elapsed_seconds=time.perf_counter() - generation_started if final_attempt else None,
        usage=usage, finish_reason=getattr(choice, 'finish_reason', None),
        reasoning_chars=len(reasoning), content_chars=len(content), modalities=modalities,
        error_type=type(error).__name__ if error else None,
        http_status=getattr(error, 'status_code', None) if error else None)


def record_request(**values):
    if os.environ.get('TOOLSTAR_INFERENCE_METRICS') != '1':
        return
    try:
        _record_request(**values)
    except Exception as exc:
        logger.warning('Request accounting failed: %s', exc)
