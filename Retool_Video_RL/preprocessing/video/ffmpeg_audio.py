"""Audio extraction with the ffmpeg binary (much faster than moviepy's per-frame reader for audio-only work)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ffmpeg_binary() -> Optional[str]:
    """FFMPEG_BINARY if set, otherwise ffmpeg on PATH."""
    configured = str(os.environ.get("FFMPEG_BINARY") or "").strip()
    if configured:
        return configured
    return shutil.which("ffmpeg")


def extract_audio_ffmpeg(
    video_path: str,
    output_path: Path,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> bool:
    """Write the audio of ``video_path`` (optionally only ``[start, end]`` seconds) as mono 16 kHz audio.

    Whisper resamples everything to 16 kHz mono, so nothing it uses is lost. Returns False when ffmpeg is
    unavailable or the input has no usable audio, so callers can fall back to the moviepy path.
    """
    binary = ffmpeg_binary()
    if not binary:
        return False
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial-{os.getpid()}{output_path.suffix}")

    command = [binary, "-nostdin", "-y", "-v", "error"]
    if start is not None and float(start) > 0:
        command += ["-ss", f"{float(start):.3f}"]
    command += ["-i", str(video_path)]
    if end is not None:
        command += ["-t", f"{max(0.001, float(end) - max(0.0, float(start or 0.0))):.3f}"]
    codec = ["-c:a", "pcm_s16le"] if output_path.suffix.lower() == ".wav" else ["-c:a", "libmp3lame", "-b:a", "64k"]
    command += ["-vn", "-ac", "1", "-ar", "16000", *codec, str(partial)]

    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        logger.warning("ffmpeg audio extraction could not start: %s", exc)
        return False
    if completed.returncode != 0 or not partial.exists() or partial.stat().st_size == 0:
        logger.info("ffmpeg audio extraction produced no audio for %s: %s", video_path, completed.stderr.strip()[-200:])
        partial.unlink(missing_ok=True)
        return False
    os.replace(partial, output_path)
    return True
