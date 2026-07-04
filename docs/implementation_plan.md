# Implementation plan

This document records the design choices in this open-source release so future
contributors and reviewers can navigate the codebase quickly.

## 1. Pipeline stages

1. **Candidate-pool preparation.** The official MRAG-Bench / Visual-RAG
   datasets are converted into a single JSONL schema (see
   `utility_mrag/data/candidate_pool.py`). Each line is one query plus
   GT/retrieved candidate images.
2. **Surrogate scoring.** `HelpfulnessScorer` formats the helpfulness prompt,
   runs a *single* generation step through the surrogate model, and returns
   the True/False logits for the first generated token. The True logit is the
   helpfulness score by default.
3. **Top-K selection.** `select_top_k` sorts deterministically (descending
   score, ties broken by original index) and returns the top-K candidates.
4. **Final generation.** The main model is fed the input image (MRAG-Bench
   only) plus the K selected candidates and the dataset-appropriate
   generation prompt.
5. **Evaluation.** Multiple-choice (MMMU-style extraction + scenario
   accuracy) for MRAG-Bench; exact-match / F1 / optional LLM-judge for
   Visual-RAG.

## 2. Key abstractions

* `BaseMultimodalModel` -- single interface for surrogate and main models;
  the rest of the pipeline never imports `transformers` directly.
* `ModelConfig` -- YAML-friendly description of a family + checkpoint; built
  via `model_config_from_yaml` and instantiated by `build_model`.
* `TrueFalseLogitExtractor` -- caches resolved `True` / `False` token ids
  and exposes a single `compute(...)` method that takes first-step logits.
  Documented multi-token fallback in `true_false_logits.py`.
* `HelpfulnessScorer` / `SurrogateSelector` -- thin wrappers that compose the
  above into a per-example call.

## 3. Provenance from the original repo

| Original (`code/MRAG-Bench`) | Cleaned (`code/utility-mrag`) | Notes |
| --- | --- | --- |
| `experiments/baseline_exps/prompt_reranker/true_false_softmax_prob.py` | `utility_mrag/scoring/true_false_logits.py` | Rewritten for clarity; documented multi-token fallback explicitly. |
| `experiments/baseline_exps/prompt_reranker/prompt_template_*.txt` | `utility_mrag/scoring/prompt_templates.py` | Inlined as Python constants; same wording. |
| `utils/model_wrappers/{qwen3vl,internvl,minicpm,gemma,ovis}_wrapper.py` | `utility_mrag/models/{qwen_vl,internvl,minicpm,gemma,ovis}.py` | Re-implemented behind the `BaseMultimodalModel` interface; no env-var workstation switching, no Chinese comments, no FP8 magic, no global PIL state. |
| `eval/score.py`, `eval/utils/automatic_extract.py` | `utility_mrag/evaluation/mrag_bench.py` | MMMU-style answer extraction + per-scenario aggregation. |
| `eval/calculate_clip_similarity.py` | `baselines/clip_rank.py` | Trimmed to a single ranking entrypoint. |
| `experiments/baseline_exps/prompt_reranker/prompt_reranker_exp.py` | `scripts/run_selection.py` + `selection/surrogate_selector.py` | Reorganised into a CLI + library. |

The following kinds of files were intentionally **not** ported:

* All `.log`, `.out`, `.err`, `.png`, `.csv`, `.tar(.gz)` artefacts.
* `analysis/`, `analysis_results/`, `tmp/`, `varco_*` experiment branches,
  failed-direction scripts, and any per-run JSONLs.
* `set_local_workstation.sh` and the `MRAG_LOCAL_WORKSTATION` env-var
  branching in the wrappers.
* Slurm sbatch scripts and `.pids/` directories.
* The original `.git` history.

## 4. Out of scope for the first OSS pass

* Assumption-validation experiments (deferred per instructions).
* The verbalised-UQ / listwise-ranking variants in
  `experiments/baseline_exps/prompt_reranker/`.
* Calflops-based FLOPs measurement is wired up as an *optional* extra in
  `scripts/profile_flops_latency.py` but is not run by default.
