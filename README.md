# ReTool-Video

<a href="https://arxiv.org/abs/2605.13228"><img src="https://img.shields.io/badge/arXiv-2605.13228-b31b1b" alt="arXiv"></a>

ReTool-Video answers questions about videos through a planner, a tool resolver, and video, audio, and retrieval tools. This release includes the inference pipeline and a Mix evaluator for MVBench, MLVU, and Video-MME.

## Quickstart

From the repository root on Linux with a CUDA-capable GPU, create the inference environment and set a local model directory. The environment pins the Python inference libraries and installs FFmpeg, the Hugging Face CLI, and the dataset extraction utilities.

```bash
conda env create -f environment.yml
conda activate retoolv
export MODEL_BASE_PATH="$PWD/models"

hf download Qwen/Qwen3.5-9B --local-dir "$MODEL_BASE_PATH/Qwen/Qwen3.5-9B"
hf download Qwen/Qwen3-VL-Embedding-2B --local-dir "$MODEL_BASE_PATH/Qwen/Qwen3-VL-Embedding-2B"
hf download Qwen/Qwen3-Embedding-0.6B --local-dir "$MODEL_BASE_PATH/Qwen/Qwen3-Embedding-0.6B"
hf download Systran/faster-whisper-large-v3 --local-dir "$MODEL_BASE_PATH/Systran/faster-whisper-large-v3"

python Retool_Video_RL/run_e2e.py /path/to/video.mp4 \
  "What happens in this video?" ./local_runs/single_video \
  --runtime-profile evaluation_local
```

The default model directory is `./models` relative to the working directory. To configure inference with an environment file, copy `Retool_Video_RL/.env.example` to `.env` in the repository root. `RESOLVER_MAX_SEARCH_ROUNDS` sets the number of model turns per resolver node that may contain a top-level Search.

GPU Whisper requires CUDA 12 cuBLAS and cuDNN 9 available to the dynamic linker. See the [Faster Whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu). Model paths and inference budgets are defined in [settings.py](Retool_Video_RL/config/settings.py).

### Optional vLLM backend

The optional vLLM backend serves the planner, resolver, captioning, and visual QA. Install vLLM 0.19.0 in a separate Python environment to keep its PyTorch and Transformers dependencies isolated. See the [vLLM GPU installation guide](https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/) for CUDA and driver requirements.

Set `RETOOL_VIDEO_VLLM_BIN` to the directory containing the installed `vllm` executable. Set `RETOOL_VIDEO_VLLM_MEDIA_ROOT` to the smallest shared directory containing both the input videos and the generated clips and frames. For the example below, place the datasets and evaluation workspace under the repository root.

```bash
python -m venv .venv-vllm
.venv-vllm/bin/python -m pip install "vllm==0.19.0"
export RETOOL_VIDEO_VLLM_BIN="$PWD/.venv-vllm/bin"
export RETOOL_VIDEO_VLLM_MEDIA_ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=0

bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json \
  --vllm
```

The launcher starts local replicas on the visible GPUs that have enough free memory. To use an existing vLLM server, pass `--vllm-urls http://127.0.0.1:8000/v1`, set `RETOOL_VIDEO_VLLM_MODEL` to the server's served model name, and configure its `--allowed-local-media-path` to match the shared media directory. The auto-launched replicas bind to loopback.

## Mix evaluation

### Prepare the datasets

```bash
export DATA_ROOT="$PWD/datasets"
hf download OpenGVLab/MVBench --repo-type dataset --local-dir "$DATA_ROOT/MVBench"
for archive in "$DATA_ROOT"/MVBench/video/*.zip; do
  unzip -nq "$archive" -d "$DATA_ROOT/MVBench/video"
done

hf download MLVU/MVLU --repo-type dataset --local-dir "$DATA_ROOT/MLVU"
hf download VLM2Vec/Video-MME --repo-type dataset --local-dir "$DATA_ROOT/Video-MME" \
  --include "videomme/*" --include "videos_chunked_*.zip"

mkdir -p "$DATA_ROOT/Video-MME/videos"
for archive in "$DATA_ROOT"/Video-MME/videos_chunked_*.zip; do
  unzip -nq "$archive" -d "$DATA_ROOT/Video-MME/videos"
done
```

For MVBench Fine-grained Pose, obtain the 320 NTU RGB+D videos separately and place them in `datasets/MVBench/video/nturgbd/`. The upstream dataset card links to the source and the required filenames. Set the dataset paths in [mix_config.json](evaluationVideo/Mix/mix_config.json). Dataset roots and the MVBench annotation path are relative to the directory containing the configuration file. MLVU and Video-MME annotation and video paths are relative to their dataset roots.

| Dataset | Set these paths |
| --- | --- |
| [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) | `dataset_root` to the video directory and `json_root` to the task annotations. |
| [MLVU](https://huggingface.co/datasets/MLVU/MVLU) | `dataset_root` to the directory containing the annotations and videos. Set `annotation_root` and `video_root` to their actual locations. |
| [Video-MME](https://huggingface.co/datasets/VLM2Vec/Video-MME) | `dataset_root` to the download directory. Check `annotation_path`, `video_id_mapping_path`, and `video_search_roots`. |

### Tool-specific checkpoints

The Quickstart checkpoints cover the planner, core visual QA, captioning, embeddings, and ASR. The following tools load additional models on demand. To enable them, download the checkpoints with `hf download <repo-id> --local-dir "$MODEL_BASE_PATH/<repo-id>"`.

| Tools | Checkpoints |
| --- | --- |
| Audio Clip QA and audio-based emotion analysis | `Qwen/Qwen2.5-Omni-3B` |
| Entity tracking | `IDEA-Research/grounding-dino-base` and `facebook/sam2.1-hiera-tiny` |
| Audio event detection | `laion/clap-htsat-fused` |
| Speaker diarization | `pyannote/speaker-diarization-community-1` |

The inference environment includes RapidOCR and ONNX Runtime for OCR. To use speaker diarization, accept the [checkpoint access terms](https://huggingface.co/pyannote/speaker-diarization-community-1), download the checkpoint as above, and install the worker dependencies in a separate Python environment:

```bash
python -m venv .venv-diarization
.venv-diarization/bin/python -m pip install "pyannote.audio==4.0.7" "pydantic-settings==2.12.0"
export DIARIZATION_PYTHON="$PWD/.venv-diarization/bin/python"
```

### Run the full three-dataset evaluation

The default configuration runs every loaded question across MVBench, MLVU, and Video-MME. It uses MVBench's video-file tasks and multiple-choice questions across the three datasets. After preparing the datasets, check the files and start the evaluation:

```bash
bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json --check-data
bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json
```

Append `--vllm` to use the configured vLLM backend or use `--parallel-samples` to limit concurrency. Progress and results accumulate across runs that share the same output paths. For an independent run, use a separate configuration with a fresh `workspaces_root`, `progress_path`, `results_path`, and `videomme_official_output_path`.

### Read accuracy by dataset and subset

The evaluator writes `Mix_result.json` by default. `summary_by_dataset` reports overall accuracy for each dataset. `subset_summaries_by_dataset` reports `count`, `correct`, and `acc` by MVBench task group, MLVU task type, and Video-MME subcategory. Accuracy is a fraction between 0 and 1, and only subsets represented in the accumulated results appear.

The Video-MME `official_breakdown` under `summary_by_dataset.videomme` also reports accuracy by duration, domain, subcategory, and task type. Its denominator counts responses from which an option can be extracted. The evaluator separately writes `VideoMME_official_output.json` in the benchmark's output format.

```bash
jq '.subset_summaries_by_dataset' evaluationVideo/Mix/Mix_result.json
```

## Running with untrusted inputs

`Python_Executor` runs model-generated Python with the permissions of the inference process. Run the evaluator with only the files and credentials it needs, or disable this tool by setting `DISABLED_TOOL_NAMES_BASE='["Python_Executor"]'`.

## Acknowledgement

We thank the maintainers of [vLLM](https://github.com/vllm-project/vllm), [Transformers](https://github.com/huggingface/transformers), [verl](https://github.com/volcengine/verl), and the [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench), [MLVU](https://github.com/JUNJIE99/MLVU), and [Video-MME](https://github.com/MME-Benchmarks/Video-MME) benchmarks.
