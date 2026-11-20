"""Optional, local ONNX text detection and recognition for OCR_Extract."""
from pathlib import Path
import threading

_ENGINE = None
_LOCK = threading.RLock()


def recognize_images(result, *, backend="rapidocr", language="en"):
    if backend == "frames":
        result.update(ocr_status="not_requested", ocr_backend="frames", model_forward_executed=False)
        return result
    if backend != "rapidocr":
        raise ValueError("OCR backend must be rapidocr or frames.")
    completed = 0
    try:
        if language.lower() not in {"en", "eng", "zh", "ch", "chi_sim", "chinese", "english", "auto"}:
            raise ValueError("The installed OCR checkpoint covers Chinese/English; another language needs a matching checkpoint.")
        from rapidocr import RapidOCR
        import rapidocr
        from config.settings import settings
        global _ENGINE
        with _LOCK:
            if _ENGINE is None:
                root = Path(rapidocr.__file__).parent / "models"
                # Explicit local paths prevent implicit network downloads during a tool call.
                paths = {"Det.model_path": root / "PP-OCRv6_det_small.onnx",
                         "Rec.model_path": root / "PP-OCRv6_rec_small.onnx",
                         "Cls.model_path": root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"}
                for path in paths.values():
                    if not path.is_file():
                        raise FileNotFoundError(f"OCR checkpoint missing: {path}; install rapidocr==3.9.2.")
                _ENGINE = RapidOCR(params={**{key: str(path) for key, path in paths.items()},
                                           "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                                           "EngineConfig.onnxruntime.inter_op_num_threads": 2})
            sources = []
            seen = set()
            for item in result.get("image_evidence", []):
                paths = item.get("source_images") or [item.get("image")]
                times = item.get("sample_times") or [item.get("timestamp")]
                for index, path in enumerate(paths):
                    if path and path not in seen:
                        seen.add(path)
                        sources.append((path, times[index] if index < len(times) else None))
            items = []
            for path, timestamp in sources[:settings.ocr_max_images]:
                output = _ENGINE(path)
                completed += 1
                texts = output.txts if output.txts is not None else []
                for text, score, box in zip(texts, output.scores, output.boxes):
                    items.append({"text": text, "confidence": float(score), "polygon": box.tolist(),
                                  "image_path": str(path), "timestamp": timestamp,
                                  "coordinate_space": "source_image_pixels"})
        result.update(ocr_status="success" if items else "no_text_detected", ocr_backend="rapidocr_onnx",
                      model_forward_executed=bool(completed), ocr_images_analyzed=completed,
                      ocr_images_omitted=max(0, len(sources)-completed), text_items=items,
                      extracted_text="\n".join(item["text"] for item in items),
                      ocr_models={"detector": "PP-OCRv6_det_small", "recognizer": "PP-OCRv6_rec_small"})
        result["result"] = ("Recognized visible text:\n" + result["extracted_text"]) if items else "No text was recognized; returned images remain available for inspection."
    except Exception as exc:
        result.update(ocr_status="backend_error", ocr_backend="rapidocr_onnx", ocr_error=str(exc),
                      model_forward_executed=bool(completed), ocr_images_analyzed=completed)
        result["warning"] = (str(result.get("warning") or "") + " OCR recognition unavailable: " + str(exc)).strip()
        result["result"] = "OCR recognition failed; image evidence remains available. " + str(exc)
    return result
