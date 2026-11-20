"""One isolated pyannote request; keeps its dependencies outside the eval environment."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import settings
from toolkit.perception_backends import diarize

if __name__ == "__main__":
    request = json.load(sys.stdin)
    settings.diarization_python = ""
    settings.perception_device = request["device"]
    # speaker_diarization_model_path derives from the same MODEL_BASE_PATH as the parent.
    if str(Path(settings.speaker_diarization_model_path).resolve()) != str(Path(request["model_path"]).resolve()):
        raise ValueError("Parent and worker diarization model paths differ.")
    output = diarize(request["video"], request["window"])
    output["backend_process_reused"] = False
    print(json.dumps(output))
