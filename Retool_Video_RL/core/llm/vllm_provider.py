"""vLLM-backed chat provider.

Drop-in replacement for the in-process ``TransformersProvider`` that talks to
one or more local vLLM OpenAI-compatible servers (continuous batching, prefix
cache, multi-GPU replicas selected by least in-flight requests).

Enable with ``TOOLSTAR_LLM_BACKEND=vllm`` and ``TOOLSTAR_VLLM_BASE_URLS``
(comma separated ``http://host:port/v1``).  The routing hook lives in
``provider.get_llm_provider``; the replica pool is started by
``evaluationVideo/Mix/vllm_pool.py``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from config.settings import settings
from core.llm import provider as _p
from core.llm.provider import OpenAIProvider
from core.llm.inference_metrics import record_request, metered_stage

logger = logging.getLogger(__name__)

_DEFAULT_VIDEO_FPS = 2.0
_PROMPT_RE = re.compile(r"(\d+)\s+(?:tokens\s+)?in the messages", re.IGNORECASE)


def vllm_backend_enabled() -> bool:
    return (
        str(os.environ.get("TOOLSTAR_LLM_BACKEND", "")).strip().lower() == "vllm"
        and bool(vllm_base_urls())
    )


def vllm_base_urls() -> str:
    return str(os.environ.get("TOOLSTAR_VLLM_BASE_URLS", "") or "").strip()


def vllm_served_model_name() -> str:
    return str(os.environ.get("TOOLSTAR_VLLM_MODEL", "") or "qwen35-9b").strip()


def _media_root() -> Path:
    return Path(os.environ.get("TOOLSTAR_VLLM_MEDIA_ROOT", str(Path.cwd()))).resolve()


def _jpeg_b64(image: Image.Image, *, max_pixels: Optional[int] = None, quality: int = 92) -> str:
    image = image.convert("RGB")
    if max_pixels:
        w, h = image.size
        if w * h > int(max_pixels):
            scale = (int(max_pixels) / float(w * h)) ** 0.5
            image = image.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class _ProcessorMarker:
    """Truthy stand-in for ``TransformersProvider.processor`` so multimodal gates stay open."""

    def __repr__(self) -> str:  # pragma: no cover
        return "<vLLM remote processor>"


_POOLS: Dict[str, Any] = {}

_LAST_REASONING = threading.local()


def pop_last_reasoning() -> str:
    """Reasoning text the server split off the last generation made on this thread (cleared on read)."""
    value = str(getattr(_LAST_REASONING, "value", "") or "")
    _LAST_REASONING.value = ""
    return value


def _extra_field(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value



def _shared_pool(base_urls: str):
    """One in-process endpoint pool shared by every role so in-flight counts are global."""
    from core.sidecar_pool import OpenAIEndpointPool

    key = ",".join(_p.split_endpoint_list(base_urls))
    pool = _POOLS.get(key)
    if pool is None:
        pool = OpenAIEndpointPool(api_key="EMPTY", base_urls=key, namespace="vllm_backend")
        _POOLS[key] = pool
    return pool


def vllm_serves_model(model_name: Any) -> bool:
    """True when ``model_name`` is the checkpoint served by the vLLM replicas."""
    if not vllm_backend_enabled() or not model_name:
        return False
    served = os.environ.get("TOOLSTAR_VLLM_MODEL_PATH") or getattr(settings, "planner_model_path", "")
    try:
        return Path(str(model_name)).expanduser().resolve() == Path(str(served)).expanduser().resolve()
    except Exception:
        return False


def get_vllm_provider(model_name: Any, model_role: Optional[str], **kwargs: Any) -> "VLLMProvider":
    return VLLMProvider(
        model_role=model_role,
        local_model_path=str(model_name),
        **{k: v for k, v in kwargs.items() if k not in {"device", "device_map", "model_name"}},
    )


class VLLMProvider(OpenAIProvider):
    """Qwen multimodal chat via local vLLM servers."""

    model = None  # no in-process weights; caption runtime sharing falls back to the vLLM visual processor
    processor = _ProcessorMarker()
    _supports_multimodal_generation = True
    provider_family = "transformers"  # generation-config / thinking semantics identical to local HF
    requires_frame_payload = True

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        model_role: Optional[str] = None,
        local_model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        explicit = {k: kwargs.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "enable_thinking", "max_tokens") if kwargs.get(k) is not None}
        # Role-specific model: the planner can be served as a vLLM LoRA module (e.g. a distilled adapter) while every other
        # role keeps the base model on the same replicas. Unset = old behaviour.
        super().__init__(
            model_name=model_name or vllm_served_model_name(),
            api_key="EMPTY",
            base_url=base_url or vllm_base_urls(),
            model_role=model_role,
        )
        self.local_model_path = str(local_model_path or "")
        defaults = _p._get_role_generation_config(model_role, "transformers")
        self.temperature = explicit.get("temperature", defaults.get("temperature", 0.7))
        self.max_tokens = int(explicit.get("max_tokens") or defaults.get("max_new_tokens") or 8192)
        self.top_p = explicit.get("top_p", defaults.get("top_p"))
        self.top_k = explicit.get("top_k", defaults.get("top_k"))
        self.min_p = explicit.get("min_p", defaults.get("min_p"))
        self.presence_penalty = explicit.get("presence_penalty", defaults.get("presence_penalty"))
        self.repetition_penalty = explicit.get("repetition_penalty", defaults.get("repetition_penalty"))
        self.enable_thinking = _p._resolve_enable_thinking_for_role(
            model_role,
            explicit.get("enable_thinking", defaults.get("enable_thinking", False)),
            fallback=bool(defaults.get("enable_thinking", False)),
            source="VLLMProvider.__init__",
            provider_name="transformers",
        )
        self._endpoint_pool = _shared_pool(base_url or vllm_base_urls())
        self.base_url = self._endpoint_pool.display_base_urls
        self._tokenizer = None
        self._server_max_len: Optional[int] = None

    # ------------------------------------------------------------------ media
    def _wire_model(self) -> str:
        """Model name sent to vLLM, decided per request from the CURRENT role. Role views (copy.copy + new model_role) reuse a
        provider that was built as "planner" for clip QA / whole-video view; those must stay on the base model, so the planner
        adapter is chosen here and not when the provider is constructed."""
        role = str(getattr(self, "model_role", "") or "").strip().lower()
        key = {"planner": "TOOLSTAR_VLLM_PLANNER_MODEL", "subagent": "TOOLSTAR_VLLM_RESOLVER_MODEL"}.get(role)
        override = str(os.environ.get(key, "") or "").strip() if key else ""
        if override:
            return override
        return self.model_name

    def _image_url(self, value: Any, *, max_pixels: Optional[int] = None) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, Image.Image):
            return "data:image/jpeg;base64," + _jpeg_b64(value, max_pixels=max_pixels)
        text = str(value).strip()
        if not text:
            return None
        if text.startswith(("http://", "https://", "data:")):
            return text
        if text.startswith("file://"):
            text = text[len("file://"):]
        path = Path(text).expanduser()
        if not path.exists():
            return None
        path = path.resolve()
        if not _p._is_readable_visual_file(path):
            return None
        if max_pixels is None:
            try:
                path.relative_to(_media_root())
                return "file://" + str(path)
            except ValueError:
                pass
        with Image.open(path) as handle:
            return "data:image/jpeg;base64," + _jpeg_b64(handle, max_pixels=max_pixels)

    def _convert_item(self, item: Dict[str, Any], extra: Dict[str, Any]) -> List[Dict[str, Any]]:
        item_type = str(item.get("type") or "").strip().lower()
        if item_type == "text":
            return [{"type": "text", "text": str(item.get("text") or "")}]
        if item_type in {"image_url", "video_url"}:
            return [item]
        if item_type == "image":
            raw = item.get("image") or item.get("path") or item.get("url")
            max_pixels = _positive_int(item.get("max_pixels"))
            if isinstance(raw, str) and not raw.startswith(("http://", "https://", "data:")):
                p = Path(raw.replace("file://", "")).expanduser()
                if p.exists() and not _p._is_readable_visual_file(p.resolve()):
                    rebuilt = _p._rebuild_visual_reference_from_sources(
                        p.resolve(), list(item.get("source_images") or [])
                    )
                    raw = rebuilt or ""
            url = self._image_url(raw, max_pixels=max_pixels)
            if not url:
                logger.warning("vLLM provider skipped unreadable image input: %r", str(raw)[:200])
                return []
            out: List[Dict[str, Any]] = []
            temporal = {k: item[k] for k in ("timestamp", "sample_times") if item.get(k) not in (None, "", [])}
            if temporal:
                out.append({
                    "type": "text",
                    "text": "Temporal metadata in seconds for the following image: "
                    + json.dumps(temporal, ensure_ascii=False, default=str),
                })
            out.append({"type": "image_url", "image_url": {"url": url}})
            return out
        if item_type == "video":
            raw = item.get("video") or item.get("path") or item.get("url")
            fps = item.get("fps") or item.get("sample_fps") or _DEFAULT_VIDEO_FPS
            try:
                fps = float(fps)
            except (TypeError, ValueError):
                fps = _DEFAULT_VIDEO_FPS
            max_pixels = _positive_int(item.get("max_pixels"))
            if isinstance(raw, (list, tuple)):
                frames: List[str] = []
                for frame in raw:
                    if frame is None:
                        continue
                    if isinstance(frame, Image.Image):
                        frames.append(_jpeg_b64(frame, max_pixels=max_pixels))
                    else:
                        url = self._image_url(frame, max_pixels=max_pixels or 10**9)
                        if url and url.startswith("data:image"):
                            frames.append(url.split(",", 1)[1])
                if not frames:
                    return []
                extra.setdefault("media_io_kwargs", {})["video"] = {"fps": fps, "num_frames": -1}
                return [{"type": "video_url", "video_url": {"url": "data:video/jpeg;base64," + ",".join(frames)}}]
            text = str(raw or "").replace("file://", "")
            if text.startswith(("http://", "https://")):
                url = text
            else:
                path = Path(text).expanduser().resolve()
                if not path.exists():
                    return []
                url = "file://" + str(path)
            # num_frames=-1: the loader keeps fps-based sampling instead of its 32-frame default,
            # which the processor's own do_sample_frames pass cannot reconcile.
            if item.get("server_sampling"):
                # The request's mm_processor_kwargs (fps, do_sample_frames) alone decide the sampling.
                return [{"type": "video_url", "video_url": {"url": url}}]
            kw: Dict[str, Any] = {"fps": fps, "num_frames": -1}
            if _positive_int(item.get("max_frames")):
                kw["num_frames"] = int(item["max_frames"])
            extra.setdefault("media_io_kwargs", {})["video"] = kw
            return [{"type": "video_url", "video_url": {"url": url}}]
        return []

    def _convert_messages(self, messages: List[Dict[str, Any]]):
        extra: Dict[str, Any] = {}
        converted: List[Dict[str, Any]] = []
        has_visual = False
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                parts: List[Dict[str, Any]] = []
                for item in content:
                    if isinstance(item, dict):
                        parts.extend(self._convert_item(item, extra))
                    elif item is not None:
                        parts.append({"type": "text", "text": str(item)})
                if any(p.get("type") in {"image_url", "video_url"} for p in parts):
                    has_visual = True
                if not parts:
                    parts = [{"type": "text", "text": ""}]
                new_message = {**message, "content": parts}
            else:
                new_message = {**message, "content": "" if content is None else str(content)}
            converted.append(new_message)
        self._cap_images(converted)
        return converted, extra, has_visual

    @staticmethod
    def _cap_images(converted: List[Dict[str, Any]]) -> int:
        """Keep the newest images when a prompt exceeds the server's per-prompt image limit.

        The vLLM server rejects prompts with more than `limit_mm_per_prompt` images (64 by default),
        which killed long agent runs whose history kept accumulating frames. Older images are
        replaced by a short text marker so message structure and ordering stay intact.
        """
        limit = _positive_int(os.environ.get("TOOLSTAR_VLLM_MAX_IMAGES")) or 60
        slots = [
            (mi, pi)
            for mi, message in enumerate(converted)
            if isinstance(message.get("content"), list)
            for pi, part in enumerate(message["content"])
            if part.get("type") == "image_url"
        ]
        drop = slots[: max(0, len(slots) - limit)]
        for mi, pi in drop:
            converted[mi]["content"][pi] = {"type": "text", "text": "[earlier image omitted]"}
        return len(drop)

    # ----------------------------------------------------------- tokenization
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            path = self.local_model_path or os.environ.get("TOOLSTAR_VLLM_MODEL_PATH") or settings.planner_model_path
            self._tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, len(text) // 4)

    @metered_stage("tokenization", backend="vllm_tokenize")
    def _server_tokenize(self, messages: List[Dict[str, Any]], extra: Dict[str, Any]) -> Optional[int]:
        with self._borrow_chat_client() as (_client, base_url):
            root = base_url.rstrip("/")
            root = root[:-3] if root.endswith("/v1") else root
            body = {"model": self._wire_model(), "messages": messages, "add_generation_prompt": True, **extra}
            req = urllib.request.Request(
                root + "/tokenize",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return int(json.loads(resp.read())["count"])

    def count_message_tokens(self, messages, **kwargs) -> int:
        try:
            converted, extra, has_visual = self._convert_messages(messages)
            if has_visual:
                n = self._server_tokenize(converted, extra)
                if n:
                    return n
            prompt = self.tokenizer.apply_chat_template(
                converted, tokenize=False, add_generation_prompt=True
            )
            return self.count_tokens(prompt)
        except Exception as exc:
            logger.debug("vLLM token count fallback: %s", exc)
            return super().count_message_tokens(messages, **kwargs)

    def _max_model_len(self) -> int:
        if self._server_max_len is None:
            try:
                with self._borrow_chat_client() as (client, _):
                    models = client.models.list().data
                    self._server_max_len = int(getattr(models[0], "max_model_len", 0) or 0)
            except Exception:
                self._server_max_len = 0
        return self._server_max_len or 65536

    def get_max_context_length(self) -> int:
        configured = int(getattr(settings, "max_input_tokens", 0) or 0)
        ceiling = max(512, self._max_model_len() - 8192)
        return min(configured, ceiling) if configured > 0 else ceiling

    # -------------------------------------------------------------- generation
    def _prepare_kwargs(self, messages, temperature, max_tokens, kwargs):
        kwargs = dict(kwargs)
        converted, extra, _ = self._convert_messages(messages)
        output_without_limit = bool(kwargs.pop("output_without_limit", False))
        if output_without_limit and self.model_role != "evolver":
            raise ValueError("Uncapped output is scoped to Evolver")
        deadline = kwargs.pop("wall_clock_deadline_monotonic", None)
        gen_deadline = kwargs.pop("generation_deadline_monotonic", None)
        kwargs.pop("max_time", None)
        if os.environ.get("RQB_DISABLE_WALL_DEADLINES") == "1":
            deadline = gen_deadline = None
        explicit_do_sample = kwargs.pop("do_sample", None)
        allowed = {
            "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty",
            "stop", "enable_thinking", "respect_role_generation_config", "extra_body", "seed",
            "timeout", "logit_bias",
        }
        dropped = sorted(k for k in kwargs if k not in allowed)
        if dropped:
            logger.debug("vLLM provider ignoring HF-only generate kwargs: %s", dropped)
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        temp = self.temperature if temperature is None else temperature
        if explicit_do_sample is None:
            pass
        elif not explicit_do_sample:
            temp = 0.0
        else:
            try:
                if temp is None or float(temp) <= 0:
                    temp = 1.0
            except Exception:
                temp = 1.0
        new_max = int(max_tokens or self.max_tokens or 8192)
        if (
            not output_without_limit
            and bool(getattr(settings, "enforce_max_output_tokens", False))
            and getattr(settings, "max_output_tokens", 0)
            and int(settings.max_output_tokens) > 0
        ):
            new_max = min(new_max, int(settings.max_output_tokens))
        if output_without_limit:
            try:
                used = self._server_tokenize(converted, extra) or self.count_message_tokens(messages)
            except Exception:
                used = self.count_message_tokens(messages)
            new_max = max(256, self._max_model_len() - int(used) - 64)
        timeouts = [d - time.monotonic() for d in (deadline, gen_deadline) if d is not None]
        if timeouts:
            kwargs["timeout"] = max(5.0, min(timeouts))
        body = dict(kwargs.pop("extra_body", {}) or {})
        for key, val in extra.items():
            body[key] = val
        kwargs["extra_body"] = body
        return converted, temp, new_max, kwargs

    def _request_payload(self, converted, temp, new_max, kw, base_url) -> Dict[str, Any]:
        kw = dict(kw)
        body = dict(kw.pop("extra_body", {}) or {})
        thinking = _p._resolve_enable_thinking_for_role(
            self.model_role,
            kw.pop("enable_thinking", None),
            fallback=bool(self.enable_thinking),
            source="VLLMProvider.generate",
            provider_name="transformers",
            respect_role_config=bool(kw.pop("respect_role_generation_config", True)),
        )
        body = _p._inject_openai_thinking_control(body, enable_thinking=thinking, base_url=base_url)
        for key, default in (("top_k", self.top_k), ("min_p", self.min_p), ("repetition_penalty", self.repetition_penalty)):
            value = kw.pop(key, default)
            if value is not None:
                body[key] = value
        payload: Dict[str, Any] = {
            "model": self._wire_model(),
            "messages": converted,
            "temperature": temp,
            "max_tokens": new_max,
            "extra_body": body,
            "stop": kw.pop("stop", None) or list(_p._SYNTHETIC_ROLE_STOP_SEQUENCES),
        }
        top_p = kw.pop("top_p", self.top_p)
        if top_p is not None:
            payload["top_p"] = top_p
        presence = kw.pop("presence_penalty", None)
        if presence is not None:
            payload["presence_penalty"] = presence
        payload.update(kw)
        if "seed" not in payload and os.environ.get("TOOLSTAR_VLLM_REQUEST_SEED"):
            payload["seed"] = int(os.environ["TOOLSTAR_VLLM_REQUEST_SEED"])
        return payload

    def generate(self, messages, temperature=None, max_tokens=None, **kwargs) -> str:
        generation_started = time.perf_counter()
        call_id = uuid.uuid4().hex
        converted, temp, new_max, kw = self._prepare_kwargs(messages, temperature, max_tokens, kwargs)
        started = time.perf_counter()
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            attempt_started = time.perf_counter()
            base_url, payload = None, {}
            try:
                with self._borrow_chat_client() as (client, base_url):
                    payload = self._request_payload(converted, temp, new_max, kw, base_url)
                    response = client.chat.completions.create(**payload)
                record_request(provider_role=self.model_role, call_id=call_id, attempt=attempt + 1,
                    attempt_started=attempt_started, generation_started=generation_started,
                    payload=payload, endpoint=base_url, response=response, final_attempt=True)
                break
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                prompt_tokens = self._overflow_prompt_tokens(exc)
                if prompt_tokens is not None:
                    room = self._max_model_len() - prompt_tokens - 16
                    if room >= 256 and room < new_max and attempt == 0:
                        logger.warning("Clamping max_tokens %d -> %d (prompt=%d)", new_max, room, prompt_tokens)
                        new_max = room
                        record_request(provider_role=self.model_role, call_id=call_id, attempt=attempt + 1,
                            attempt_started=attempt_started, generation_started=generation_started,
                            payload=payload, endpoint=base_url, error=exc)
                        continue
                try:
                    self._raise_prompt_overflow(exc)
                except Exception:
                    record_request(provider_role=self.model_role, call_id=call_id, attempt=attempt + 1,
                        attempt_started=attempt_started, generation_started=generation_started,
                        payload=payload, endpoint=base_url, error=exc, final_attempt=True)
                    raise
                retryable = status in (408, 429, 500, 502, 503, 504) or exc.__class__.__name__ in {
                    "APIConnectionError", "APITimeoutError"
                }
                last_exc = exc
                record_request(provider_role=self.model_role, call_id=call_id, attempt=attempt + 1,
                    attempt_started=attempt_started, generation_started=generation_started,
                    payload=payload, endpoint=base_url, error=exc, final_attempt=not retryable or attempt == 2)
                if not retryable or attempt == 2:
                    logger.error("vLLM request failed role=%s: %s", self.model_role, str(exc)[:500])
                    raise
                time.sleep(2.0 * (attempt + 1))
        else:  # pragma: no cover
            raise last_exc  # type: ignore[misc]
        choice = response.choices[0]
        text = choice.message.content or ""
        logger.info(
            "vLLM generation finished: role=%s duration=%.3fs prompt_tokens=%s completion_tokens=%s finish=%s",
            self.model_role or "unknown", time.perf_counter() - started,
            getattr(response.usage, "prompt_tokens", None), getattr(response.usage, "completion_tokens", None),
            choice.finish_reason,
        )
        if choice.finish_reason == "length":
            logger.warning("Generation likely hit max_tokens limit (%d)", new_max)
        returned = _p._sanitize_decoded_text(text)
        try:
            _LAST_REASONING.value = str(
                _extra_field(choice.message, "reasoning") or _extra_field(choice.message, "reasoning_content") or ""
            )
        except Exception:  # noqa: BLE001 - reasoning text is optional
            _LAST_REASONING.value = ""
        return returned

    def stream_generate(self, messages, temperature=None, max_tokens=None, **kwargs):
        converted, temp, new_max, kw = self._prepare_kwargs(messages, temperature, max_tokens, kwargs)
        return super().stream_generate(converted, temperature=temp, max_tokens=new_max, **kw)

    @staticmethod
    def _overflow_prompt_tokens(exc: Exception) -> Optional[int]:
        text = str(exc)
        if "maximum context length" not in text:
            return None
        match = _PROMPT_RE.search(text)
        return int(match.group(1)) if match else None

    def _raise_prompt_overflow(self, exc: Exception) -> None:
        text = str(exc)
        if "maximum context length" not in text and "maximum input length" not in text:
            return
        match = _PROMPT_RE.search(text)
        tokens = int(match.group(1)) if match else 0
        limit = self.get_max_context_length()
        raise ValueError(
            f"Prompt requires {tokens} tokens, exceeding configured max_input_tokens="
            f"{limit}. Silent truncation is disabled; reduce context or enable memory folding."
        ) from exc

    def close(self) -> None:
        return None
