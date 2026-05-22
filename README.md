# ReTool-Video

ReTool-Video is an agent framework for video understanding and tool use. The current open-source version mainly organizes and preserves the directly runnable Mix evaluation pipeline, which is used to evaluate a model's video tool-use capability on `MVBench`, `MLVU`, and `Video-MME`.

## Repository Structure

```text
ReTool-Video/
├─ README.md
├─ environment.yml
├─ docs/
│  ├─ README.md
│  ├─ SETUP.md
│  ├─ PRD.md
│  ├─ Implementation.md
│  ├─ TODO.md
│  ├─ RL_solution.md
│  └─ conclusion.md
├─ evaluationVideo/
│  └─ Mix/
│     ├─ run_mix_batch.py
│     ├─ run_mix_batch.sh
│     ├─ mix_config.json
│     ├─ mix_datasets.py
│     ├─ mix_state.py
│     ├─ reward_metrics.py
│     └─ *_query_prompt.txt
└─ Retool_Video_RL/
   ├─ run_e2e.py
   ├─ config/
   ├─ planner/
   ├─ preprocessing/
   ├─ runtime/
   ├─ toolkit/
   └─ core/
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate retoolv
```

## Models and Configuration

The configuration is in [Retool_Video_RL/config/settings.py](Retool_Video_RL/config/settings.py).

Before running, you need to confirm at least the following:

1. `model_base_path`
   The current default value is `/mnt/data/models`. If your model directory is different, change it to the actual local path on your machine.
2. Model subpaths
   For example, `planner_model_subpath`, `subagent_model_subpath`, and `qwen3_vl_embedding_model_subpath`. By default, they point to local Qwen-series models.
3. `.env` or environment variables
   `Settings` supports reading configuration from `.env`. If you use an OpenAI-compatible service, you can also provide variables such as `API_KEY`, `API_BASE`, and `MODEL`.

If you are simply following the current default evaluation pipeline, the top priority is to make sure the local model paths are correct.

## Dataset and Model Download Guide

The default Mix configuration file is [evaluationVideo/Mix/mix_config.json](evaluationVideo/Mix/mix_config.json). Before running, you need to prepare the datasets and local models first, and then modify the paths in this configuration file according to your machine's directory layout.

### 0. Install the Hugging Face CLI First

`hf download` is the recommended download method from the official Hugging Face CLI. See the official documentation for command details:
<https://huggingface.co/docs/huggingface_hub/package_reference/cli>

```bash
python -m pip install -U "huggingface_hub[cli]"
hf auth login
```

Some datasets or models require you to accept the license or request access on the Hugging Face page first, especially:

- `OpenGVLab/MVBench`
- `MLVU/MVLU`
- `pyannote/speaker-diarization-community-1`

### 1. Dataset Download Commands

The following commands are written based on the default directory structure of the current repository.

```bash
export DATA_ROOT=/path/to/datasets
mkdir -p "$DATA_ROOT"
```

#### 1.1 MVBench

```bash
hf download OpenGVLab/MVBench \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/MVBench"
```

References:

- Paper: <https://arxiv.org/abs/2311.17005>
- Hugging Face: <https://huggingface.co/datasets/OpenGVLab/MVBench>

#### 1.2 MLVU

```bash
hf download MLVU/MVLU \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/MLVU"
```

References:

- Paper: <https://arxiv.org/abs/2406.04264>
- Official repository: <https://github.com/JUNJIE99/MLVU>
- Hugging Face Dev Set: <https://huggingface.co/datasets/MLVU/MVLU>

#### 1.3 Video-MME

```bash
hf download VLM2Vec/Video-MME \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/Video-MME" \
  --include "videomme/*" \
  --include "videos_chunked_*.zip"

mkdir -p "$DATA_ROOT/Video-MME/videos"
for f in "$DATA_ROOT"/Video-MME/videos_chunked_*.zip; do
  unzip -q "$f" -d "$DATA_ROOT/Video-MME/videos"
done
```

References:

- Official repository: <https://github.com/MME-Benchmarks/Video-MME>
- Version including compressed video files: <https://huggingface.co/datasets/VLM2Vec/Video-MME>

### 2. Model Download Commands

First, set the local model root directory. This directory needs to match the `model_base_path` in [Retool_Video_RL/config/settings.py](Retool_Video_RL/config/settings.py).

```bash
export MODEL_ROOT=/path/to/models
mkdir -p "$MODEL_ROOT"
```

#### 2.1 Minimum Recommended Downloads for the Current Mix Evaluation

These are the most essential models in the current uploaded Mix evaluation version and the ones most likely to be triggered in practice:

```bash
hf download Qwen/Qwen3.5-9B \
  --local-dir "$MODEL_ROOT/Qwen/Qwen3.5-9B"

hf download Qwen/Qwen3-VL-Embedding-2B \
  --local-dir "$MODEL_ROOT/Qwen/Qwen3-VL-Embedding-2B"

hf download Qwen/Qwen3-Embedding-0.6B \
  --local-dir "$MODEL_ROOT/Qwen/Qwen3-Embedding-0.6B"

hf download Systran/faster-whisper-large-v3 \
  --local-dir "$MODEL_ROOT/Systran/faster-whisper-large-v3"
```

These correspond to:

- `Qwen/Qwen3.5-9B`
  - Default local model for planner / subagent / preprocess / `Video_Clip_QA`
- `Qwen/Qwen3-VL-Embedding-2B`
  - Embedding model for preprocessing vector retrieval
- `Qwen/Qwen3-Embedding-0.6B`
  - `Tool_Search` embedding
- `Systran/faster-whisper-large-v3`
  - ASR

#### 2.2 Recommended Additional Downloads for Full Tool Coverage

If you want to avoid discovering missing models only after a certain tool is triggered during evaluation, it is recommended to download the following as well:

```bash
hf download Qwen/Qwen3.5-0.8B \
  --local-dir "$MODEL_ROOT/Qwen/Qwen3.5-0.8B"

hf download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen/Qwen2.5-Coder-7B-Instruct"

hf download IDEA-Research/grounding-dino-base \
  --local-dir "$MODEL_ROOT/IDEA-Research/grounding-dino-base"

hf download laion/clap-htsat-fused \
  --local-dir "$MODEL_ROOT/laion/clap-htsat-fused"

hf download pyannote/speaker-diarization-community-1 \
  --local-dir "$MODEL_ROOT/pyannote/speaker-diarization-community-1"

hf download speechbrain/emotion-recognition-wav2vec2-IEMOCAP \
  --local-dir "$MODEL_ROOT/speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

hf download Xenova/facial_emotions_image_detection \
  --local-dir "$MODEL_ROOT/Xenova/facial_emotions_image_detection"

hf download briaai/RMBG-2.0 \
  --local-dir "$MODEL_ROOT/briaai/RMBG-2.0"

hf download iic/cv_rrdb_image-super-resolution \
  --local-dir "$MODEL_ROOT/iic/cv_rrdb_image-super-resolution"
```

These cover:

- `Qwen/Qwen3.5-0.8B`
  - `Query_Rewrite`, `Fold_Memory`
- `Qwen/Qwen2.5-Coder-7B-Instruct`
  - Code assistance path for `Python_Executor`
- `IDEA-Research/grounding-dino-base`
  - `Entity_Track`
- `laion/clap-htsat-fused`
  - `Audio_Event_Detect`
- `pyannote/speaker-diarization-community-1`
  - `Speaker_Diarization`
- `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`
  - Audio emotion recognition
- `Xenova/facial_emotions_image_detection`
  - Image emotion recognition
- `briaai/RMBG-2.0`
  - `Crop`
- `iic/cv_rrdb_image-super-resolution`
  - `Zoom_in`

### 3. Path Check After Downloading

Before actual use, you usually need to check both of the following:

- The dataset paths in `evaluationVideo/Mix/mix_config.json`
- The `model_base_path` in [Retool_Video_RL/config/settings.py](Retool_Video_RL/config/settings.py)

A typical dataset directory layout looks like this:

```text
datasets/
├─ MVBench/
│  ├─ video/
│  └─ json/
├─ MLVU/
│  └─ MLVU/
│     ├─ json/            # or upload_json/
│     └─ video/
└─ Video-MME/
   ├─ videomme/
   │  └─ test-00000-of-00001.parquet
   ├─ videos/
   └─ data/
```

## Start Evaluation

### 1. Run the Full Mix Evaluation

Linux / macOS:

```bash
bash evaluationVideo/Mix/run_mix_batch.sh evaluationVideo/Mix/mix_config.json
```

Common arguments:

```bash
python evaluationVideo/Mix/run_mix_batch.py \
  --config evaluationVideo/Mix/mix_config.json \
  --max-samples 10 \
  --parallel-samples 2 \
  --skip-true
```

Argument descriptions:

- `--max-samples`: Only run the next N pending samples
- `--parallel-samples`: Number of parallel samples; by default automatically inferred from visible CUDA devices
- `--no-cache`: Disable preprocessing cache reuse
- `--skip-true`: Import existing correct results into the progress file first to avoid rerunning them

### 2. Debug a Single Sample

You can directly call [Retool_Video_RL/run_e2e.py](Retool_Video_RL/run_e2e.py):

```bash
python Retool_Video_RL/run_e2e.py \
  /path/to/video.mp4 \
  "Answer the question about this video." \
  ./debug_workspace \
  --runtime-profile evaluation_local
```

Common optional arguments:

- `--no-cache`
- `--log-path`
- `--no-file-log`
- `--flat-workspace`
- `--shared-preprocess-root`
- `--runtime-profile evaluation_local`

Among them, `evaluation_local` forces the use of the local evaluation runtime configuration defined in the current codebase, which is suitable for offline evaluation scenarios.

## Runtime Outputs

After the Mix evaluation finishes, the following files are generated by default:

- `evaluationVideo/Mix/Mix_result.json`
  The aggregated final results, including `summary_by_dataset`, `subset_summaries_by_dataset`, and per-sample results.
- `evaluationVideo/Mix/VideoMME_official_output.json`
  Output for compatibility with the official `Video-MME` format.
- `evaluationVideo/Mix/workspaces/mix_progress.json`
  Progress file for resuming from checkpoints.
- `evaluationVideo/Mix/workspaces/.../mix_sample_result.json`
  Result file for each individual sample.
- `evaluationVideo/Mix/workspaces/.../launcher.log`
  Launch log for each individual sample.

By default, the repository ignores most runtime artifacts, for example:

- `evaluationVideo/**/workspaces*/`
- `evaluationVideo/**/_shared_cache/`
- `Retool_Video_RL/videorag_cache/`
- `*.log`

## Notes

- The current README corresponds to the **w/o RL** evaluation version. The `Tool-use` metrics already reported in the current paper are `74.6 / 65.3 / 73.3`. Related scripts and documentation, including RL training, will continue to be added and uploaded later.

## Acknowledgement

This project benefited from the following open-source project during implementation and organization, and we sincerely appreciate its support:

- `verl`: <https://github.com/volcengine/verl>
