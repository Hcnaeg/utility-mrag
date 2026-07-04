# utility-mrag

**[ACL 2026] Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation**

`utility-mrag` is the open-source release of our paper. It implements a
*surrogate-accelerated* pipeline for multimodal RAG: a small surrogate
multimodal LLM scores each candidate image by its **helpfulness** -- the
final-layer logit of the `True` token under the prompt *"Is this image helpful
for answering the query?"*. The Top-K most helpful candidates are then handed
to a larger main model for final answer generation.

```
+--------------+     helpfulness         +--------------+      Top-K
|  candidates  | ---- score (True logit) |  surrogate   | -------------+
|  (per qid)   |     <one fwd / cand>    |  Qwen3-VL-2B |              |
+--------------+                         +--------------+              v
                                                                +--------------+
                                                                | main model   |
                                                                | (e.g. 8B)    |
                                                                | generates    |
                                                                | final answer |
                                                                +--------------+
```

The repo supports both **MRAG-Bench** (multiple-choice, with an input image
plus retrieved evidence) and **Visual-RAG** (open-ended, retrieval-only).

## Installation

This project is managed with [`uv`](https://github.com/astral-sh/uv).

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync the environment from pyproject.toml + uv.lock
uv sync

# Optional extras:
uv sync --extra clip       # OpenCLIP baselines
uv sync --extra eval       # Visual-RAG LLM-as-judge
uv sync --extra profile    # FLOPs profiling
uv sync --extra datasets   # HuggingFace datasets loader
uv sync --extra dev        # ruff, mypy, pytest

# 3. Run tests to confirm the install
uv run pytest tests
```

A toy quickstart that doesn't require any model weights:

```bash
uv run python examples/quickstart_selection.py
```

## Data preparation

Raw datasets are not bundled. See [`data/README.md`](data/README.md) and
[`docs/data_preparation.md`](docs/data_preparation.md) for download links and
preparation commands. The prepare scripts produce a JSONL manifest per
dataset:

```bash
uv run python scripts/prepare_mrag_bench.py \
    --input_dir /path/to/mrag_bench \
    --retrieval_file /path/to/retrieved_candidates.jsonl \
    --output data/manifests/mrag_bench_candidates.jsonl

uv run python scripts/prepare_visual_rag.py \
    --input_dir /path/to/visual_rag \
    --retrieval_file /path/to/retrieved_candidates.jsonl \
    --output data/manifests/visual_rag_candidates.jsonl
```

## End-to-end pipeline

### 1. Score candidates with the surrogate and pick Top-K

```bash
uv run python scripts/run_selection.py \
    --manifest data/manifests/mrag_bench_candidates.jsonl \
    --surrogate_config configs/models/qwen3_vl_2b_surrogate.yaml \
    --top_k 1 3 5 \
    --output_dir outputs/selection/mrag_bench/qwen3_vl_2b
```

This emits `top1.jsonl`, `top3.jsonl`, `top5.jsonl` plus a full `all_scores.jsonl`.

### 2. Generate answers with the main model

```bash
uv run python scripts/run_generation.py \
    --selection_file outputs/selection/mrag_bench/qwen3_vl_2b/top3.jsonl \
    --question_image_lookup data/manifests/mrag_bench_candidates.jsonl \
    --main_model_config configs/models/qwen3_vl_8b.yaml \
    --output outputs/generation/mrag_bench/qwen3_vl_8b_top3.jsonl
```

### 3. Evaluate

```bash
uv run python scripts/run_eval.py \
    --dataset mrag_bench \
    --pred outputs/generation/mrag_bench/qwen3_vl_8b_top3.jsonl \
    --output outputs/eval/mrag_bench/qwen3_vl_8b_top3.json
```

For Visual-RAG free-form answers, optionally enable the LLM-as-judge:

```bash
export OPENAI_API_KEY=...    # required; never hard-code in repo
uv run python scripts/run_eval.py \
    --dataset visual_rag \
    --pred outputs/generation/visual_rag/qwen3_vl_8b_top3.jsonl \
    --use_llm_judge --judge_model gpt-4o-mini \
    --output outputs/eval/visual_rag/qwen3_vl_8b_top3.json
```

## Baselines

The `baselines/` package ports the comparators from the paper:

```bash
# CLIP / SigLIP / OpenCLIP image-text similarity ranking
uv run python -m baselines.clip_rank --manifest ... --output_dir ...
uv run python -m baselines.siglip_rank --manifest ... --output_dir ...
uv run python -m baselines.openclip_rank --manifest ... --output_dir ...

# Answer-level uncertainty (Table 2)
uv run python -m baselines.answer_level_uq \
    --manifest ... --main_model_config configs/models/qwen3_vl_8b.yaml \
    --uq_method softmax_entropy --output_dir ...
```

## Reproducing the main tables

See [`docs/reproduction.md`](docs/reproduction.md) for full table-by-table
commands and [`docs/expected_results.md`](docs/expected_results.md) for
target accuracies.

## Model wrappers

All wrappers conform to a single interface
([`utility_mrag/models/base.py`](utility_mrag/models/base.py)): supply a
`ModelConfig` (or YAML) and get back a model that exposes
`score_true_false_logits(...)` and `generate_answer(...)`.

Built-in wrappers cover Qwen3-VL, MiniCPM-V 4.5, Gemma 3, Ovis 2.5, and
InternVL 3.5. Add a new wrapper by subclassing `BaseMultimodalModel` and
decorating with `@register_model("my_family")`.

See [`docs/model_setup.md`](docs/model_setup.md) for installation notes
(flash-attention, AWQ, HF tokens for gated models).

## Layout

```
utility-mrag/
├── utility_mrag/                core library
│   ├── scoring/                 helpfulness scoring + True/False logits
│   ├── selection/               Top-K selection
│   ├── models/                  per-family wrappers
│   ├── data/                    candidate-pool I/O
│   └── evaluation/              MRAG-Bench MCQ + Visual-RAG metrics
├── scripts/                     CLI entrypoints (uv run python scripts/...)
├── baselines/                   CLIP / SigLIP / OpenCLIP / answer-UQ
├── configs/                     YAML configs (datasets / models / experiments)
├── data/                        manifests (raw datasets are NOT shipped)
├── examples/                    quickstart + toy candidate pool
├── tests/                       pytest unit tests (no model weights needed)
└── docs/                        reproduction / data prep / model setup
```

## Notes on licenses and external datasets

* This repo is released under the MIT License (see `LICENSE`).
* MRAG-Bench, Visual-RAG, and the various model checkpoints are subject to
  their **own** licenses. Please consult each upstream source before use.
* `utility-mrag` ships **no** raw images, model weights, or API keys.

## Citation

```bibtex
@inproceedings{luo2026utility,
  title={Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation},
  author={Luo, Weiqing and Hu, Zongye and Wang, Xiao and Yu, Zhiyuan and Zhang, Haofeng and Huang, Ziyi},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={35091--35124},
  year={2026}
}
```
