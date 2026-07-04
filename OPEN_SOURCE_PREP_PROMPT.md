# Open-Source Code Preparation Prompt for Claude Code

You are helping me clean and prepare a research codebase for open-source release.

## Context

- The original messy research repository is located at: `code/MRAG-Bench`
- The new clean open-source repository has already been created at: `code/utility-mrag`
- The paper is titled: **“Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation”**
- The clean repo should contain only the code needed to reproduce the main method and experiments, not the full messy research history.
- Do **not** copy the original `.git` directory, logs, raw data dumps, cache files, model checkpoints, temporary notebooks, failed research directions, or unrelated scripts.
- Assumption validation logic can be skipped for now. Do **not** implement assumption-validation scripts in this pass.
- Prioritize implementing the main open-source pipeline.

## Python Environment and Dependency Management

Important environment details:

- My current local development environment is a conda environment named `qwen3vl`.
- You may use this environment to inspect, test, and run existing code locally.
- However, the open-source version of the repo should use **uv** for dependency management.
- Prepare the clean repo as a modern Python project managed by `uv`.

Dependency-management requirements:

- Create and maintain a clean `pyproject.toml`.
- Use `uv` as the primary installation and execution method.
- Generate `uv.lock` if possible.
- Do not rely on conda-specific environment files for the open-source artifact.
- Do not make `requirements.txt` the primary dependency source.
- It is acceptable to include an optional `requirements.txt` only for compatibility, but `pyproject.toml` + `uv.lock` should be the canonical setup.
- In README and docs, show commands using `uv`, for example:

```bash
uv sync
uv run python scripts/run_selection.py --help
uv run pytest tests
````

* If some dependencies are GPU/CUDA-specific or model-specific, document them clearly as optional extras in `pyproject.toml` when reasonable.
* Do not hard-code environment paths from the `qwen3vl` conda environment.

## Goal

Prepare `code/utility-mrag` as a clean, well-structured open-source artifact for the paper.

The repo should support:

1. Building candidate pools for MRAG-Bench and Visual-RAG.
2. Scoring candidate images using latent-helpfulness probing.
3. Extracting True/False token logits from surrogate multimodal models.
4. Ranking candidate evidence and selecting Top-K images.
5. Running final answer generation with a main multimodal model.
6. Evaluating results on MRAG-Bench and Visual-RAG.
7. Running key baselines needed for comparison, especially CLIP/SigLIP-style ranking and answer-level uncertainty baselines if existing code is available.
8. Providing clean configs, scripts, README, and documentation for reproducibility.
9. Managing the open-source Python project with `uv`.

## Important Method Details

The core method estimates visual evidence utility through a latent helpfulness variable.

For each query `q` and candidate image `c_i`, construct an auxiliary prompt similar to:

```text
Is this image helpful for answering the query?
```

The output space is binary:

```text
True / False
```

The helpfulness score should be based on the final-layer logit of the `"True"` token, or a clean equivalent implementation if tokenization requires model-specific handling.

Candidate images are ranked by this helpfulness score.

The Top-K selected images are passed to the main model for final answer generation.

This is a surrogate-accelerated pipeline: a lightweight surrogate model scores all candidates, and the larger main model is only used for final generation.

## Tasks

### 1. Inspect the Original Repository

Explore `code/MRAG-Bench`.

Identify files related to:

* dataset preparation
* MRAG-Bench processing
* Visual-RAG processing
* candidate pool construction
* model wrappers for Qwen3-VL, MiniCPM, Gemma, Ovis, InternVL
* True/False logit extraction
* helpfulness / utility scoring
* Top-K image selection
* generation with selected evidence
* evaluation scripts
* CLIP/SigLIP/OpenCLIP retrieval or ranking baselines
* answer-level uncertainty baselines, if already implemented
* FLOPs/latency profiling, if already implemented cleanly

### 2. Create a Clean Repository Structure

Create this target structure in `code/utility-mrag`:

```text
utility-mrag/
  README.md
  LICENSE
  CITATION.cff
  pyproject.toml
  uv.lock

  utility_mrag/
    __init__.py
    scoring/
      __init__.py
      helpfulness_score.py
      true_false_logits.py
      prompt_templates.py
    selection/
      __init__.py
      topk.py
      surrogate_selector.py
    models/
      __init__.py
      base.py
      qwen_vl.py
      minicpm.py
      gemma.py
      ovis.py
      internvl.py
    data/
      __init__.py
      candidate_pool.py
      dataset_loaders.py
    evaluation/
      __init__.py
      mrag_bench.py
      visual_rag.py
      llm_judge.py
      metrics.py

  scripts/
    prepare_mrag_bench.py
    prepare_visual_rag.py
    build_candidate_pool.py
    run_selection.py
    run_generation.py
    run_eval.py
    profile_flops_latency.py

  baselines/
    __init__.py
    clip_rank.py
    siglip_rank.py
    openclip_rank.py
    answer_level_uq.py

  configs/
    datasets/
      mrag_bench.yaml
      visual_rag.yaml
    models/
      qwen3_vl_2b_surrogate.yaml
      qwen3_vl_8b.yaml
      minicpm_v45.yaml
      gemma3_12b.yaml
      ovis25_9b.yaml
      internvl35_8b.yaml
    experiments/
      table1_mrag.yaml
      table1_visual_rag.yaml
      table2_answer_level.yaml
      table4_efficiency.yaml

  data/
    README.md
    manifests/
      .gitkeep

  examples/
    quickstart_selection.py
    toy_candidate_pool.jsonl

  tests/
    test_true_false_logits.py
    test_topk_selection.py
    test_candidate_pool.py

  docs/
    implementation_plan.md
    reproduction.md
    data_preparation.md
    model_setup.md
    expected_results.md
```

### 3. Move or Rewrite Only Relevant Code

Copy useful logic from `code/MRAG-Bench` into `code/utility-mrag`, but clean it up.

Requirements:

* Remove hard-coded local paths.
* Replace private absolute paths with CLI arguments or YAML config fields.
* Remove unrelated experiment branches.
* Remove debug-only code and dead code.
* Remove personal comments, temporary TODOs, and machine-specific assumptions.
* Preserve the core algorithmic behavior.
* Keep APIs simple and documented.
* Do not copy large files, logs, raw datasets, checkpoints, generated outputs, or hidden cache directories.

### 4. Set Up `uv` Project Metadata

Create a clean `pyproject.toml`.

Suggested baseline:

```toml
[project]
name = "utility-mrag"
version = "0.1.0"
description = "Utility-oriented visual evidence selection for multimodal retrieval-augmented generation."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
  { name = "Weiqing Luo" }
]
dependencies = [
  "torch",
  "transformers",
  "accelerate",
  "pillow",
  "numpy",
  "pandas",
  "tqdm",
  "pyyaml",
  "jsonlines",
  "scikit-learn",
  "pytest",
]

[project.optional-dependencies]
dev = [
  "pytest",
  "ruff",
  "mypy",
]
clip = [
  "open-clip-torch",
]
eval = [
  "openai",
]
profile = [
  "fvcore",
]

[tool.uv]
package = true

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Adjust dependencies based on the actual imported packages from the cleaned code.

After creating `pyproject.toml`, run when possible:

```bash
uv sync
uv lock
uv run pytest tests
```

If `uv` is not available in the current shell, still prepare `pyproject.toml` correctly and document the expected commands.

### 5. Implement the Main Pipeline

The clean repo should expose these main workflows.

#### A. Candidate Pool Preparation

Expected command examples:

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

Candidate manifest format should be JSONL, one example per line:

```json
{
  "qid": "...",
  "query": "...",
  "candidate_images": [
    {"image_id": "...", "image_path": "...", "source": "gt"},
    {"image_id": "...", "image_path": "...", "source": "retrieved"}
  ],
  "gt_image_ids": ["..."],
  "answer": "...",
  "metadata": {}
}
```

#### B. Helpfulness Scoring and Top-K Selection

Expected command:

```bash
uv run python scripts/run_selection.py \
  --manifest data/manifests/mrag_bench_candidates.jsonl \
  --surrogate_config configs/models/qwen3_vl_2b_surrogate.yaml \
  --top_k 1 3 5 \
  --output_dir outputs/selection/mrag_bench/qwen3_vl_2b
```

The output should include JSONL files with selected images and scores, for example:

```json
{
  "qid": "...",
  "query": "...",
  "top_k": 3,
  "selected_images": [
    {
      "image_id": "...",
      "image_path": "...",
      "score": 12.34,
      "true_logit": 12.34,
      "false_logit": 8.91
    }
  ],
  "all_scores": []
}
```

#### C. Final Generation

Expected command:

```bash
uv run python scripts/run_generation.py \
  --selection_file outputs/selection/mrag_bench/qwen3_vl_2b/top3.jsonl \
  --main_model_config configs/models/qwen3_vl_8b.yaml \
  --output outputs/generation/mrag_bench/qwen3_vl_8b_top3.jsonl
```

#### D. Evaluation

Expected command:

```bash
uv run python scripts/run_eval.py \
  --dataset mrag_bench \
  --pred outputs/generation/mrag_bench/qwen3_vl_8b_top3.jsonl \
  --output outputs/eval/mrag_bench/qwen3_vl_8b_top3.json
```

For MRAG-Bench, implement exact-match or the existing metric used in the original repo.

For Visual-RAG, implement the existing LLM-as-Judge evaluation if available, but make it optional and clearly documented. API keys must come from environment variables, never hard-coded files.

### 6. Model Wrapper Design

Implement a clean model abstraction in:

```text
utility_mrag/models/base.py
```

Suggested interface:

```python
class BaseMultimodalModel:
    def format_helpfulness_input(self, query: str, image_path: str):
        raise NotImplementedError

    def score_true_false_logits(self, query: str, image_path: str) -> dict:
        raise NotImplementedError

    def generate_answer(self, query: str, image_paths: list[str]) -> str:
        raise NotImplementedError
```

Each model wrapper should implement the methods as needed.

If some model wrappers from the original repo are incomplete or too messy, create clean placeholders with clear `NotImplementedError` messages and documentation, but fully implement the wrappers that are already available and working.

Model families to support/configure:

* Qwen3-VL
* MiniCPM-V4.5
* Gemma3
* Ovis2.5
* InternVL3.5

### 7. True/False Token Logit Extraction

Implement this carefully in:

```text
utility_mrag/scoring/true_false_logits.py
```

Requirements:

* Provide a model-agnostic helper where possible.
* Handle tokenization differences.
* Check whether `"True"` and `"False"` are single tokens for each tokenizer.
* If they are multiple tokens, document and implement a reasonable fallback, such as first-token logit or summed/mean logit over token pieces.
* Return both `true_logit` and `false_logit`.
* The default helpfulness score should be `true_logit`.
* Add tests for the extraction logic using a mock tokenizer/logit tensor.

### 8. Prompt Templates

Implement prompt templates in:

```text
utility_mrag/scoring/prompt_templates.py
```

Include at least:

* default helpfulness prompt
* alternative paraphrased prompt if already used in the paper experiments
* generation prompt template

Keep templates concise and easy to modify.

### 9. Top-K Selection

Implement in:

```text
utility_mrag/selection/topk.py
```

Requirements:

* Sort candidates by helpfulness score descending.
* Support arbitrary `k`.
* Preserve full score records.
* Make tie-breaking deterministic.
* Add unit tests.

### 10. Baselines

Create clean baseline scripts if reusable code exists:

* CLIP ranking
* OpenCLIP ranking
* SigLIP ranking
* answer-level uncertainty baseline

Do not spend too much time reimplementing unavailable baselines from scratch. If the original repo has working code, clean and port it. If not, add documented placeholders and mark them as optional.

### 11. FLOPs / Latency Profiling

If clean profiling logic exists in the original repo, port it into:

```text
scripts/profile_flops_latency.py
```

This is lower priority than the main pipeline. Do not block the main implementation on profiling.

### 12. Documentation

Write a strong `README.md` with:

* project title
* short method description
* installation with `uv`
* data preparation
* quickstart with toy data
* running selection
* running generation
* running evaluation
* reproducing main tables
* model setup
* expected input/output formats
* notes about licenses and external datasets
* citation section

Use `uv` commands in documentation, for example:

```bash
uv sync
uv run python examples/quickstart_selection.py
uv run python scripts/run_selection.py --help
uv run pytest tests
```

Also write:

* `docs/reproduction.md`
* `docs/data_preparation.md`
* `docs/model_setup.md`
* `docs/expected_results.md`
* `data/README.md`

`data/README.md` should explain that raw datasets are not included and users must download them from official sources.

### 13. Safety and Privacy Cleanup

Before finishing:

* Ensure no `.git` from the old repo is copied.
* Ensure no raw large datasets are copied into the clean repo.
* Ensure no model checkpoints are copied.
* Ensure no logs, cache, wandb folders, slurm outputs, or temporary files are copied.
* Ensure no API keys or tokens are present.
* Ensure no absolute personal paths remain.
* Ensure no private server names or usernames remain.
* Ensure the open-source repo is not dependent on the local conda environment `qwen3vl`.
* Ensure `uv` is the documented dependency manager.

Add a `.gitignore` that excludes:

```gitignore
.env
*.log
*.out
*.err
__pycache__/
.ipynb_checkpoints/
wandb/
outputs/
checkpoints/
data/raw/
data/cache/
*.pt
*.pth
*.safetensors
```

Run a final grep for risky strings:

```bash
grep -R "api_key\|OPENAI\|HF_TOKEN\|/home/\|/scratch/\|/mnt/\|wandb\|slurm\|password\|secret\|token" .
```

Review any matches carefully. Some matches may be harmless documentation references, but no actual secrets or private paths should remain.

### 14. Tests and Sanity Checks

Add lightweight tests:

* test True/False logit extraction with mocked logits
* test Top-K ranking
* test candidate pool JSONL parsing

Run:

```bash
uv run pytest tests
```

Also run a toy quickstart using:

```bash
uv run python examples/quickstart_selection.py
```

If `uv` is unavailable in the shell, document that tests should be run with these commands once `uv` is installed.

### 15. Final Output

At the end, summarize:

* what files were created
* what logic was ported from the original repo
* what logic was newly written
* what remains as placeholders
* how to run the main pipeline with `uv`
* any missing dependencies or manual setup needed
* any assumptions you made
* whether `uv sync`, `uv lock`, and `uv run pytest tests` succeeded

Do not implement assumption-validation logic in this pass. Focus on the main open-source code pipeline.

```
```
