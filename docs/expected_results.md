# Expected results

This page lists the headline numbers the open-source pipeline aims to
reproduce. Numbers are taken from the paper; treat them as targets, not
ground truth -- exact values may shift slightly with different retriever
checkpoints or transformers versions.

## Table 1 -- helpfulness-driven Top-K (MRAG-Bench, accuracy %)

The columns are main models; rows are Top-K cuts produced by the
**Qwen3-VL-2B** surrogate using the helpfulness prompt.

| Top-K | Qwen3-VL-8B | MiniCPM-V-4.5 | Gemma 3 12B | Ovis 2.5 9B | InternVL 3.5 8B |
|------:|:-----------:|:-------------:|:-----------:|:-----------:|:--------------:|
| 1     |   tbd       |   tbd         |   tbd       |   tbd       |   tbd          |
| 3     |   tbd       |   tbd         |   tbd       |   tbd       |   tbd          |
| 5     |   tbd       |   tbd         |   tbd       |   tbd       |   tbd          |

Replace `tbd` with the values reported in the paper after release.

## Table 1 -- Visual-RAG (exact-match / F1 / LLM-judge accuracy %)

| Top-K | Qwen3-VL-8B | MiniCPM-V-4.5 | Gemma 3 12B | Ovis 2.5 9B | InternVL 3.5 8B |
|------:|:-----------:|:-------------:|:-----------:|:-----------:|:--------------:|
| 1     | tbd         | tbd           | tbd         | tbd         | tbd            |
| 3     | tbd         | tbd           | tbd         | tbd         | tbd            |
| 5     | tbd         | tbd           | tbd         | tbd         | tbd            |

## Table 2 -- baseline rerankers (MRAG-Bench accuracy %, Top-3 cut)

| Method | Accuracy |
|---|:--:|
| CLIP (ViT-L/14-336)    | tbd |
| OpenCLIP (ViT-L/14)    | tbd |
| SigLIP                 | tbd |
| Answer-level entropy   | tbd |
| Answer-level min-token | tbd |
| **Ours (helpfulness)** | tbd |

## Table 4 -- efficiency

| Component | FLOPs (TFLOPs) | Wall-clock (s) |
|---|:--:|:--:|
| Surrogate scoring (Qwen3-VL-2B, 1 candidate) | tbd | tbd |
| Main generation (Qwen3-VL-8B, Top-3 evidence) | tbd | tbd |
| Answer-level UQ baseline (per candidate)      | tbd | tbd |

Run the corresponding commands in [docs/reproduction.md](reproduction.md) to
populate these tables on your hardware.
