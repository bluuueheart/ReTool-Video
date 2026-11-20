# ReTool-Video

ReTool-Video answers questions about videos through a planner, a tool resolver, and video, audio, and retrieval tools. This release contains the inference pipeline and the Mix evaluator used for the rebuttal-stage evaluation on MVBench, MLVU, and Video-MME. The workflow runs with the base Qwen3.5-9B checkpoint. A fresh run computes its opening video observation directly from the input video.

## Quickstart

From the repository root on Linux with a CUDA-capable GPU, create the inference environment and set a local model directory. The environment includes FFmpeg and the Transformers backend.

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

The default model directory is `./models` relative to the working directory. To configure inference with an environment file, copy `Retool_Video_RL/.env.example` to `.env` in the repository root. `RESOLVER_MAX_SEARCH_ROUNDS` sets the number of model turns per resolver node that may contain a top-level Search. The legacy `SUBAGENT_MAX_ITERATIONS` name remains accepted.

Use `hf auth login` for gated checkpoints and datasets. If your CTranslate2 installation cannot use the available CUDA libraries, set `PREPROCESS_ASR_DEVICE=cpu` to run Whisper on the CPU. Model paths and inference budgets are defined in [settings.py](Retool_Video_RL/config/settings.py).

### Optional vLLM backend

The rebuttal-stage runs used vLLM for the planner, resolver, captioning, and visual QA. Install upstream vLLM in a separate Python environment because its Transformers requirements may differ from the inference client's. Follow [vLLM's GPU installation guide](https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/) and the [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B).

Set `TOOLSTAR_VLLM_BIN` to the directory containing the installed `vllm` executable. Set `TOOLSTAR_VLLM_MEDIA_ROOT` to the smallest shared directory containing both the input videos and the generated clips and frames. For the example below, place the datasets and evaluation workspace under the repository root.

```bash
export TOOLSTAR_VLLM_BIN=/path/to/vllm-env/bin
export TOOLSTAR_VLLM_MEDIA_ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=0

bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json \
  --vllm --max-samples 3
```

The launcher starts local replicas on the visible GPUs that have enough free memory. To use an existing vLLM server, pass `--vllm-urls http://127.0.0.1:8000/v1`, set `TOOLSTAR_VLLM_MODEL` to the server's served model name, and configure its `--allowed-local-media-path` to match the shared media directory. The auto-launched replicas bind to loopback.

## Mix evaluation

### Prepare the datasets

```bash
export DATA_ROOT="$PWD/datasets"
hf download OpenGVLab/MVBench --repo-type dataset --local-dir "$DATA_ROOT/MVBench"
hf download MLVU/MVLU --repo-type dataset --local-dir "$DATA_ROOT/MLVU"
hf download VLM2Vec/Video-MME --repo-type dataset --local-dir "$DATA_ROOT/Video-MME" \
  --include "videomme/*" --include "videos_chunked_*.zip"

mkdir -p "$DATA_ROOT/Video-MME/videos"
for archive in "$DATA_ROOT"/Video-MME/videos_chunked_*.zip; do
  unzip -q "$archive" -d "$DATA_ROOT/Video-MME/videos"
done
```

Extract any MVBench or MLVU video archives so the dataset paths point to playable files. Set the dataset paths in [mix_config.json](evaluationVideo/Mix/mix_config.json). Dataset roots and the MVBench annotation path are relative to the directory containing the configuration file. MLVU and Video-MME annotation and video paths are relative to their dataset roots.

| Dataset | Set these paths |
| --- | --- |
| [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) | `dataset_root` to the video directory and `json_root` to the task annotations. |
| [MLVU Dev](https://huggingface.co/datasets/MLVU/MVLU) | `dataset_root` to the directory containing the annotations and videos. Set `annotation_root` and `video_root` to their actual locations. |
| [Video-MME](https://huggingface.co/datasets/VLM2Vec/Video-MME) | `dataset_root` to the download directory. Check `annotation_path`, `video_id_mapping_path`, and `video_search_roots`. |

Keep model checkpoints under `MODEL_BASE_PATH` in their Hugging Face repository layout. Additional tools load their checkpoints on demand. See [settings.py](Retool_Video_RL/config/settings.py) for the paths used by audio QA, entity detection and tracking, emotion recognition, and image transforms. For speaker diarization, download `pyannote/speaker-diarization-community-1` and install `pyannote.audio`. To use a separate environment for diarization, set `DIARIZATION_PYTHON` to its Python executable.

### Run all enabled datasets

```bash
bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json
```

Append `--vllm` to use the configured vLLM backend. Use `--max-samples` for a quick check or `--parallel-samples` to limit concurrency. Progress and results accumulate across runs that share the same output paths. For an independent run, use a separate configuration with a fresh `workspaces_root`, `progress_path`, `results_path`, and `videomme_official_output_path`.

### Run a stratified sample

The sampler allocates questions by subset size within each dataset, selects samples whose videos are available, and balances Video-MME durations within those allocations.

```bash
python evaluationVideo/Mix/make_stratified_sample.py \
  --config evaluationVideo/Mix/mix_config.json \
  --counts mvbench=33,mlvu=33,videomme=34 \
  --out-dir ./local_runs/stratified

bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json \
  --sample-uids @./local_runs/stratified/uids.txt
```

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
