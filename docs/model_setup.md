# Model setup

`utility-mrag` ships **no** model weights. The wrappers under
`utility_mrag/models/` know how to drive each family but you must download
the checkpoints yourself.

## Supported families

| Family | Default checkpoint | Notes |
| --- | --- | --- |
| `qwen3_vl` | `Qwen/Qwen3-VL-{2B,4B,8B}-Instruct` (or the `8B-FP8` variant) | Surrogate uses the 2B; main model uses the 8B. |
| `minicpm` | `openbmb/MiniCPM-V-4_5` (and `-AWQ`) | AWQ requires the `awq` package. |
| `gemma` | `google/gemma-3-{4b,12b}-it` | Gated; set `HF_TOKEN`. |
| `ovis` | `AIDC-AI/Ovis2.5-{2B,9B}` | May require an HF token. |
| `internvl` | `OpenGVLab/InternVL3_5-{2B,8B}` | Uses `trust_remote_code=True`. |

All wrappers are registered with the `@register_model("...")` decorator and
can be selected purely from a YAML config; see `configs/models/*.yaml`.

## Authentication

Gated models read the token from `HF_TOKEN` or `HUGGINGFACE_TOKEN` (in that
order). API keys are *never* read from a file in this repo:

```bash
export HF_TOKEN=hf_xxx                # for Gemma 3, Ovis 2.5 etc.
export OPENAI_API_KEY=sk-xxx          # only if running the Visual-RAG judge
```

## Optional accelerators

* **flash-attention 2** -- recommended for Qwen3-VL on Ampere / Hopper.
  Install with `pip install flash-attn --no-build-isolation` (or via your
  vendor's wheel index). Set `attn_implementation: flash_attention_2`
  in the model YAML.
* **AWQ quantisation** -- only for `openbmb/MiniCPM-V-4_5-AWQ`. Install the
  `awq` package separately; the wrapper switches to
  `AutoAWQForCausalLM.from_quantized` automatically.
* **calflops** -- enables FLOPs measurement in
  `scripts/profile_flops_latency.py`. Install via `uv sync --extra profile`.

## Adding a new model family

1. Create `utility_mrag/models/my_family.py`.
2. Subclass `BaseMultimodalModel` and implement `load`, `tokenizer`,
   `score_true_false_logits`, and `generate_answer`.
3. Decorate the class with `@register_model("my_family")`.
4. Drop a YAML under `configs/models/` and reference it from your scripts.

The True/False extraction code already handles tokenizer quirks generically;
unless your tokenizer encodes "True"/"False" in an exotic way you do not need
to touch `utility_mrag/scoring/true_false_logits.py`.
