"""Local raw-media backends. Missing dependencies/weights never yield proxy evidence."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from config.settings import settings
from runtime.session import runtime_session
from toolkit.model_results import normalize_model_result
from toolkit.time_utils import parse_time_range

_MODELS = {}
_LOCK = threading.RLock()
_INFERENCE = threading.local()


def _completed_model_call():
    _INFERENCE.completed = getattr(_INFERENCE, "completed", 0) + 1


class BackendUnavailable(RuntimeError):
    pass


def device():
    return settings.perception_device or runtime_session.get_stable_device("perception")


def model_path(subpath):
    path = Path(settings.model_base_path) / subpath
    if not (path / "config.json").is_file():
        raise BackendUnavailable(f"Local model is unavailable: {path}")
    return str(path.resolve())


def load_model(kind, path):
    import torch
    from transformers import AutoProcessor

    target = device()
    key = (kind, path, target)
    with _LOCK:
        if key not in _MODELS:
            if not Path(path).is_dir():
                raise BackendUnavailable(f"Local model is unavailable: {path}")
            if kind == "clap":
                from transformers import ClapModel
                cls = ClapModel
            elif kind == "detector":
                from transformers import AutoModelForZeroShotObjectDetection
                cls = AutoModelForZeroShotObjectDetection
            elif kind == "tracker":
                from transformers import Sam2VideoModel
                cls = Sam2VideoModel
            elif kind == "omni":
                from transformers import Qwen2_5OmniThinkerForConditionalGeneration
                cls = Qwen2_5OmniThinkerForConditionalGeneration
            else:
                raise ValueError(f"Unknown backend {kind}")
            kwargs = {"local_files_only": True}
            if kind == "omni":
                kwargs.update(dtype=torch.bfloat16 if target.startswith("cuda") else torch.float32)
            model = cls.from_pretrained(path, **kwargs).to(target).eval()
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            _MODELS[key] = (model, processor, target)
    return _MODELS[key]


def resolve_video(explicit=None):
    selected = explicit or getattr(runtime_session, "_active_video_path", None)
    if not selected or not Path(selected).is_file():
        raise ValueError("Supply video_path or set an active video; raw-media tools do not search other videos.")
    return str(Path(selected).resolve())


def resolve_window(video, time_range=None):
    import cv2
    cap = cv2.VideoCapture(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    finally:
        cap.release()
    window = parse_time_range(time_range) if time_range is not None else (0., duration)
    if not window or not 0 <= window[0] < window[1] <= duration + .05:
        raise ValueError(f"Use a positive absolute window inside video duration {duration:.3f}s.")
    if window[1] - window[0] > settings.perception_max_window_seconds:
        raise ValueError(f"Requested {window[1]-window[0]:.1f}s exceeds raw-media backend limit {settings.perception_max_window_seconds}s; locate a shorter window first.")
    return window[0], min(window[1], duration)


def source_metadata(video, window):
    path = Path(video)
    stat = path.stat()
    identity = hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    return {"source_video": str(path), "source_identity": identity, "start_time": window[0], "end_time": window[1], "time_basis": "absolute_seconds"}


@contextmanager
def audio_waveform(video, window, sampling_rate):
    import numpy as np
    from preprocessing.video.ffmpeg_audio import ffmpeg_binary
    binary = ffmpeg_binary()
    if not binary:
        raise BackendUnavailable("ffmpeg is required to decode raw audio.")
    # Decode at the actual backend rate; CLAP and Whisper/Omni have different rates.
    command = [binary, "-nostdin", "-v", "error", "-ss", str(window[0]), "-i", video,
               "-t", str(window[1]-window[0]), "-vn", "-ac", "1", "-ar", str(sampling_rate), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode or not result.stdout:
        raise ValueError("The requested clip has no usable audio: " + result.stderr.decode(errors="replace")[-300:])
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    yield waveform


def sample_frames(video, window, fps=4.):
    import cv2
    import numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(video)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        if native_fps <= 0 or frame_count <= 0:
            raise ValueError("Video has no usable frame rate/count metadata.")
        first = int(np.ceil(window[0]*native_fps-1e-7))
        last = min(frame_count-1, int(np.ceil(window[1]*native_fps-1e-7))-1)
        if first > last:
            raise ValueError("No source frame timestamp lies inside the requested window.")
        count = min(settings.perception_max_frames, max(2, int((window[1]-window[0])*fps)), last-first+1)
        indices = np.linspace(first, last, count, dtype=int).tolist()
        times = [index/native_fps for index in indices]
        for index, timestamp in zip(indices, times):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not decode frame at {timestamp:.3f}s")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return frames, times


def audio_events(video, window, description, window_seconds=5., stride_seconds=2.5):
    import torch
    path = model_path("laion/clap-htsat-fused")
    model, processor, target = load_model("clap", path)
    rate = int(processor.feature_extractor.sampling_rate)
    labels = [description, "silence", "ordinary speech", "background music"]
    items = []
    with _LOCK, torch.inference_mode(), audio_waveform(video, window, rate) as waveform:
        for local_start in range(0, len(waveform), max(1, int(rate*stride_seconds))):
            clip = waveform[local_start:local_start+int(rate*window_seconds)]
            if len(clip) < rate*.1:
                continue
            inputs = processor(text=labels, audio=[clip], sampling_rate=rate, padding=True, return_tensors="pt").to(target)
            scores = model(**inputs).logits_per_audio[0].float().cpu().tolist()
            _completed_model_call()
            items.append({"start": window[0]+local_start/rate, "end": min(window[1], window[0]+(local_start+len(clip))/rate),
                          "event": description, "match_score": scores[0], "comparison_scores": dict(zip(labels, scores)),
                          "top_candidate": labels[max(range(len(scores)), key=scores.__getitem__)]})
    return {"status": "success", "backend": "clap_raw_audio", "model_path": path, "model_forward_executed": True,
            "sampling_rate": rate, "items": items, "score_kind": "uncalibrated_text_audio_similarity",
            "result": f"Scored {len(items)} raw-audio windows; use candidate scores and intervals, not a probability or exact onset claim."}


def audio_qa(video, window, query, include_video=False):
    import torch
    path = model_path(settings.audio_qa_model_subpath)
    model, processor, target = load_model("omni", path)
    rate = int(processor.feature_extractor.sampling_rate)
    images, times = sample_frames(video, window, fps=.5) if include_video else ([], [])
    # Bound images separately from tracking; timestamps stay in the prompt.
    if len(images) > 12:
        indices = [round(i*(len(images)-1)/11) for i in range(12)]
        images, times = [images[i] for i in indices], [times[i] for i in indices]
    content = [{"type": "audio", "audio": "raw_clip.wav"}]
    content += [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": f"Question: {query}\nClip absolute window {window}; image timestamps {times}. Return JSON with answer, reasoning (evidence-based explanation), evidence, and uncertainty. Distinguish a heard voice from identifying who sings. Do not invent timestamps or speakers."})
    messages = [{"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
                {"role": "user", "content": content}]
    with _LOCK, torch.inference_mode(), audio_waveform(video, window, rate) as waveform:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, audio=[waveform], images=images or None, sampling_rate=rate, return_tensors="pt", padding=True)
        inputs = inputs.to(target)
        generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1,
                                   eos_token_id=processor.tokenizer.eos_token_id,
                                   pad_token_id=processor.tokenizer.pad_token_id,
                                   return_dict_in_generate=True)
        _completed_model_call()
        output = generated.sequences
        raw = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    result = normalize_model_result(raw)
    result.update(status="success", backend="qwen_omni_raw_audio_video" if include_video else "qwen_omni_raw_audio",
                  model_path=path, model_forward_executed=True, sampling_rate=rate, sample_times=times,
                  input_modalities=["audio", "images"] if images else ["audio"],
                  timestamp_offset_seconds=window[0], model_evidence_timestamps_unverified=True,
                  generated_tokens=int(output.shape[1]-inputs["input_ids"].shape[1]))
    if result["generated_tokens"] >= 1024:
        result["warning"] = "Model output reached its generation limit; raw response is preserved and evidence timestamps remain unverified."
    return result


def _box_iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def _associate_detections(boxes, visible, next_id):
    """Conservative spatial association; missing/ambiguous IDs are never asserted identical."""
    matches, events, used = {}, [], set()
    for box in boxes:
        ranked = sorted(((_box_iou(box, prior), oid) for oid, prior in visible.items() if oid not in used), reverse=True)
        if ranked and ranked[0][0] >= .3 and (len(ranked) == 1 or ranked[0][0]-ranked[1][0] >= .1):
            oid = ranked[0][1]
            kind = "spatially_matched"
        else:
            oid, next_id = next_id, next_id+1
            kind = "new_or_reidentified"
        used.add(oid)
        matches[oid] = box
        events.append({"object_id": oid, "association": kind,
                       "possible_previous_ids": [other for score, other in ranked if score >= .1]})
    return matches, events, next_id


def entity_track(video, window, description, threshold=.3, initial_boxes=None, redetect_interval_seconds=1.):
    import numpy as np
    import torch
    if not 0 < redetect_interval_seconds <= 10 or not 0 <= threshold <= 1:
        raise ValueError("Use a redetection interval in (0, 10] seconds and a confidence threshold in [0, 1].")
    frames, times = sample_frames(video, window)
    detector_path = model_path("IDEA-Research/grounding-dino-base")
    tracker_path = model_path(settings.tracking_model_subpath)
    with _LOCK, torch.inference_mode():
        detector, detector_processor, detector_target = load_model("detector", detector_path)
        def detect(frame):
            inputs = detector_processor(images=frame, text=description.rstrip(".")+".", return_tensors="pt").to(detector_target)
            output = detector(**inputs)
            _completed_model_call()
            detected = detector_processor.post_process_grounded_object_detection(output, inputs.input_ids, threshold=threshold,
                        text_threshold=threshold, target_sizes=[(frame.height, frame.width)])[0]
            clipped = []
            for box in detected["boxes"].float().cpu().tolist():
                if not all(np.isfinite(box)):
                    continue
                x1, y1, x2, y2 = box
                box = [max(0., x1), max(0., y1), min(float(frame.width), x2), min(float(frame.height), y2)]
                if box[0] < box[2] and box[1] < box[3]:
                    clipped.append(box)
            return clipped
        boxes = detect(frames[0]) if initial_boxes is None else initial_boxes
        for box in boxes:
            if len(box) != 4 or not all(np.isfinite(box)) or not (0 <= box[0] < box[2] <= frames[0].width and 0 <= box[1] < box[3] <= frames[0].height):
                raise ValueError("initial_boxes must be finite positive pixel xyxy boxes inside the source frame.")
        tracker, processor, target = load_model("tracker", tracker_path)
        session = processor.init_video_session(video=[np.asarray(im) for im in frames], inference_device=target,
                    video_storage_device="cpu", inference_state_device=target, dtype=torch.float32)
        object_ids = list(range(1, len(boxes)+1))
        if boxes:
            processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=object_ids, input_boxes=[boxes])
        tracks, detection_events = [], []
        next_id, last_detection = len(boxes)+1, times[0]
        uncertain_ids, lost_ids = set(), set()

        def read_masks(output, index):
            masks = processor.post_process_masks([output.pred_masks], original_sizes=[[frames[index].height, frames[index].width]], binarize=True)[0]
            current = {}
            scores = output.object_score_logits
            for offset, (oid, mask) in enumerate(zip(output.object_ids, masks)):
                pixels = mask.squeeze().detach().cpu().numpy() > 0
                ys, xs = np.where(pixels)
                present = len(xs) and (scores is None or float(scores[offset].reshape(-1)[0]) > 0)
                current[int(oid)] = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] if present else None
            return current

        for index, timestamp in enumerate(times):
            current = {}
            if session.get_obj_num():
                output = tracker(session, frame_idx=index)
                _completed_model_call()
                current = read_masks(output, index)
            if index and (timestamp-last_detection >= redetect_interval_seconds or index == len(times)-1):
                detected = detect(frames[index])
                matched, events, next_id = _associate_detections(detected, {oid: box for oid, box in current.items() if box and oid not in lost_ids}, next_id)
                for event in events:
                    event["timestamp"] = timestamp
                    if event["association"] == "new_or_reidentified" and object_ids:
                        uncertain_ids.add(event["object_id"])
                        event["identity_uncertain"] = True
                        event["lost_previous_ids"] = sorted(lost_ids)
                detection_events.extend(events)
                if matched:
                    processor.add_inputs_to_inference_session(session, frame_idx=index,
                        obj_ids=list(matched), input_boxes=[list(matched.values())])
                    object_ids = sorted(set(object_ids) | set(matched))
                    output = tracker(session, frame_idx=index)
                    _completed_model_call()
                    current = read_masks(output, index)
                last_detection = timestamp
            for oid, bbox in current.items():
                if bbox is None:
                    lost_ids.add(oid)
                elif oid in lost_ids:
                    uncertain_ids.add(oid)
                tracks.append({"object_id": oid, "timestamp": timestamp, "bbox_xyxy": bbox,
                               "state": "mask_present" if bbox else "lost_or_occluded",
                               "identity_uncertain": oid in uncertain_ids})
    return {"status": "success" if tracks else "no_valid_result", "backend": "grounding_dino_sam2", "models": [detector_path, tracker_path],
            "model_forward_executed": True, "sample_times": times, "tracks": tracks, "initial_boxes": boxes,
            "detection_events": detection_events, "redetect_interval_seconds": redetect_interval_seconds,
            "track_id_count": len(object_ids), "unique_physical_object_count": None,
            "result": f"Produced {len(tracks)} track observations with periodic target detection; track IDs are not a verified unique-object count.",
            "limitations": "Spatial association is heuristic. New IDs may be reappearances; identity after loss is uncertain. Targets visible only between sampled/detection frames can be missed. Motion is not camera-compensated."}


def diarize(video, window):
    if settings.diarization_python and os.path.abspath(settings.diarization_python) != os.path.abspath(sys.executable):
        worker = Path(__file__).with_name("diarization_worker.py")
        request = {"video": video, "window": window, "device": device(), "model_path": settings.speaker_diarization_model_path}
        process = subprocess.run([settings.diarization_python, str(worker)], input=json.dumps(request),
                                 text=True, capture_output=True)
        if process.returncode:
            raise BackendUnavailable("Isolated diarization worker failed: " + process.stderr[-1500:])
        result = json.loads(process.stdout.strip().splitlines()[-1])
        if result.get("model_forward_executed"):
            _completed_model_call()
        return result
    import torch
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise BackendUnavailable("Install pyannote.audio and obtain the diarization checkpoint before using Speaker_Diarization.") from exc
    path = Path(settings.speaker_diarization_model_path)
    if not path.exists():
        raise BackendUnavailable(f"Diarization checkpoint unavailable at {path}; Hugging Face user conditions may need acceptance.")
    key = ("diarization", str(path), device())
    with _LOCK:
        if key not in _MODELS:
            _MODELS[key] = Pipeline.from_pretrained(str(path)).to(torch.device(device()))
        with audio_waveform(video, window, 16000) as wave:
            output = _MODELS[key]({"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": 16000})
            _completed_model_call()
        annotation = getattr(output, "speaker_diarization", output)
        items = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start, end = max(window[0], window[0]+turn.start), min(window[1], window[0]+turn.end)
            if start < end:
                items.append({"start": start, "end": end, "speaker": speaker,
                              "raw_start": window[0]+turn.start, "raw_end": window[0]+turn.end})
    return {"status": "success", "backend": "pyannote_raw_audio", "model_path": str(path), "model_forward_executed": True,
            "items": items, "boundary_policy": "clipped_to_requested_window",
            "result": f"Separated {len(items)} speech turns; speaker labels are anonymous, not visual identities."}


def run_backend(operation, *, video_path=None, time_range=None, **kwargs):
    started = time.perf_counter()
    _INFERENCE.completed = 0
    metadata = {}
    try:
        video = resolve_video(video_path)
        window = resolve_window(video, time_range)
        metadata = source_metadata(video, window)
        functions = {"audio_events": audio_events, "audio_qa": audio_qa, "entity_track": entity_track, "diarize": diarize}
        result = functions[operation](video, window, **kwargs)
        return {**metadata, **result, "model_forward_call_count": _INFERENCE.completed,
                "backend_duration_seconds": time.perf_counter()-started}
    except Exception as exc:
        return {**metadata, "status": "error", "backend": operation, "model_forward_executed": _INFERENCE.completed > 0,
                "model_forward_call_count": _INFERENCE.completed,
                "error_type": "backend_unavailable" if isinstance(exc, BackendUnavailable) else "backend_error",
                "error": str(exc), "result": f"{operation} failed: {exc}", "backend_duration_seconds": time.perf_counter()-started}
