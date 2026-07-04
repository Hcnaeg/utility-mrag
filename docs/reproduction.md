# Reproducing the paper tables

All commands assume you have already run `uv sync` and prepared the candidate
pool manifests under `data/manifests/`. Outputs land under `outputs/<table>/`.

## Table 1: surrogate-driven Top-K (MRAG-Bench + Visual-RAG)

```bash
# 1. Score with the surrogate (Qwen3-VL-2B). One run per dataset.
uv run python scripts/run_selection.py \
    --manifest data/manifests/mrag_bench_candidates.jsonl \
    --surrogate_config configs/models/qwen3_vl_2b_surrogate.yaml \
    --top_k 1 3 5 \
    --output_dir outputs/table1_mrag/selection

uv run python scripts/run_selection.py \
    --manifest data/manifests/visual_rag_candidates.jsonl \
    --surrogate_config configs/models/qwen3_vl_2b_surrogate.yaml \
    --top_k 1 3 5 \
    --output_dir outputs/table1_visual_rag/selection

# 2. Generate answers with each main model (Qwen, MiniCPM, Gemma, Ovis, InternVL).
for cfg in qwen3_vl_8b minicpm_v45 gemma3_12b ovis25_9b internvl35_8b; do
  uv run python scripts/run_generation.py \
      --selection_file outputs/table1_mrag/selection/top3.jsonl \
      --question_image_lookup data/manifests/mrag_bench_candidates.jsonl \
      --main_model_config configs/models/${cfg}.yaml \
      --output outputs/table1_mrag/generation/${cfg}_top3.jsonl
done

# 3. Score.
for cfg in qwen3_vl_8b minicpm_v45 gemma3_12b ovis25_9b internvl35_8b; do
  uv run python scripts/run_eval.py --dataset mrag_bench \
      --pred outputs/table1_mrag/generation/${cfg}_top3.jsonl \
      --output outputs/table1_mrag/eval/${cfg}_top3.json
done
```

## Table 2: answer-level uncertainty baselines

```bash
# Softmax-entropy uncertainty
uv run python -m baselines.answer_level_uq \
    --manifest data/manifests/mrag_bench_candidates.jsonl \
    --main_model_config configs/models/qwen3_vl_8b.yaml \
    --uq_method softmax_entropy --top_k 1 3 5 \
    --output_dir outputs/table2/answer_uq_softmax_entropy

# Min-token-probability uncertainty
uv run python -m baselines.answer_level_uq \
    --manifest data/manifests/mrag_bench_candidates.jsonl \
    --main_model_config configs/models/qwen3_vl_8b.yaml \
    --uq_method min_token_probability --top_k 1 3 5 \
    --output_dir outputs/table2/answer_uq_min_token
```

## Table 4: efficiency profile

```bash
uv run python scripts/profile_flops_latency.py \
    --model_config configs/models/qwen3_vl_2b_surrogate.yaml \
    --task scoring --num_iters 5 \
    --output outputs/table4/qwen3vl_2b_scoring.json

uv run python scripts/profile_flops_latency.py \
    --model_config configs/models/qwen3_vl_8b.yaml \
    --task generation --num_iters 5 \
    --output outputs/table4/qwen3vl_8b_generation.json
```

(FLOPs require `uv sync --extra profile` to install `calflops`. Without it the
script reports wall-clock latency only.)

## Reranker baselines (CLIP / SigLIP / OpenCLIP)

```bash
uv run python -m baselines.clip_rank \
    --manifest data/manifests/mrag_bench_candidates.jsonl \
    --output_dir outputs/baselines/clip --top_k 1 3 5
uv run python -m baselines.siglip_rank \
    --manifest data/manifests/mrag_bench_candidates.jsonl \
    --output_dir outputs/baselines/siglip --top_k 1 3 5
uv run python -m baselines.openclip_rank \
    --manifest data/manifests/mrag_bench_candidates.jsonl \
    --output_dir outputs/baselines/openclip --top_k 1 3 5
```

OpenCLIP needs `uv sync --extra clip`.
