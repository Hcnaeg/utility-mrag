# Data preparation

This guide explains how to assemble the inputs that
`scripts/prepare_mrag_bench.py` and `scripts/prepare_visual_rag.py` consume.

## Common preconditions

Both scripts expect:

1. A local copy of the **questions** in either Hugging Face cache format or a
   plain JSONL file.
2. A **retrieval JSONL** that maps every `qid` to a ranked list of retrieved
   image paths -- one line per query::

       {"qid": "q-001", "retrieved": [{"image_id": "...", "image_path": "..."}, ...]}

   The retriever you use (CLIP / SigLIP / BGE-Vis / etc.) is up to you;
   the candidate pool itself is retriever-agnostic.

The combined output is the unified manifest documented in [data/README.md](../data/README.md).

## MRAG-Bench (zero-config, recommended)

The official `uclanlp/MRAG-Bench` HuggingFace dataset already bundles, for every
question, the input image, the ground-truth images, and the officially
CLIP-retrieved candidate images (all inline). `scripts/prepare_mrag_bench_hf.py`
extracts them straight into the unified manifest -- no local image corpus or
retrieval JSONL required:

```bash
uv sync --extra datasets     # provides `datasets` / `pyarrow`
uv run python scripts/prepare_mrag_bench_hf.py \
    --output data/manifests/mrag_bench_candidates.jsonl \
    --image_dir data/images/mrag_bench \
    --split test
```

Useful flags: `--limit N` (first N questions, for a smoke test),
`--num_candidates K` (keep at most K retrieved images per question), and
`--no_gt` (exclude ground-truth images from the candidate pool).

## MRAG-Bench (bring-your-own-retriever)

Use this path if you want to rerank candidates produced by your **own**
retriever over a local image corpus.

```bash
# 1. Download the questions and image corpus (see the upstream README).
#    The `images/` folder must be on disk.

# 2. (Optional) export questions to a JSONL with the relevant fields:
#       qid, question, A, B, C, D, scenario, gt_choice,
#       question_image_path (relative or absolute),
#       gt_image_paths (list of absolute or relative paths).
#    If you skip step 2 the prepare script falls back to the HF datasets API.

# 3. Run the prepare script.
uv run python scripts/prepare_mrag_bench.py \
    --input_dir /local/path/to/mrag_bench \
    --retrieval_file /local/path/to/retrieved_candidates.jsonl \
    --output data/manifests/mrag_bench_candidates.jsonl \
    --image_root /local/path/to/mrag_bench/images
```

Use `--qids_file` to restrict to a particular split, e.g. for the per-fold
subsets shipped with the original MRAG-Bench release.

## Visual-RAG

```bash
# 1. Convert the official Visual-RAG release to JSONL with these per-line keys:
#       qid, question, answer, category,
#       gt_image_paths (list).
#    Save as `<input_dir>/visual_rag_test.jsonl`.

# 2. Build the candidate pool.
uv run python scripts/prepare_visual_rag.py \
    --input_dir /local/path/to/visual_rag \
    --retrieval_file /local/path/to/retrieved_candidates.jsonl \
    --output data/manifests/visual_rag_candidates.jsonl \
    --image_root /local/path/to/visual_rag/images
```

## Tips

* All paths inside the manifest are stored verbatim. Use `--image_root` to
  resolve relative retrieval paths into absolute ones at preparation time so
  later stages don't depend on a particular working directory.
* If you change retrievers, regenerate only the retrieval JSONL and re-run
  the prepare script. The downstream selection / generation / eval stages are
  retriever-agnostic.
* The manifest is the *only* on-disk artifact this repo expects. Keep it
  under `data/manifests/`; it is small (kilobytes per query) and safe to
  commit if you wish to version a particular candidate pool snapshot.
