"""CLIP-based candidate ranking.

Encodes the query (text + optional input image) and each candidate image with
a Hugging Face CLIP model, then ranks candidates by cosine similarity.

Usage::

    uv run python -m baselines.clip_rank \\
        --manifest data/manifests/mrag_bench_candidates.jsonl \\
        --model openai/clip-vit-large-patch14-336 \\
        --top_k 1 3 5 \\
        --output_dir outputs/baselines/clip
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utility_mrag.data.candidate_pool import load_candidate_pool
from utility_mrag.selection.topk import select_top_k

logger = logging.getLogger("clip_rank")


def _load_clip(model_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def _query_features(
    model, processor, query: str, question_image: Image.Image | None, device: str
) -> torch.Tensor:
    inputs = processor(
        text=[query],
        images=[question_image] if question_image is not None else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_kwargs = {
            k: v for k, v in inputs.items() if k.startswith("input_ids") or k.startswith("attention_mask")
        }
        text_features = F.normalize(model.get_text_features(**text_kwargs), p=2, dim=1)
        if question_image is not None:
            img_kwargs = {k: v for k, v in inputs.items() if k.startswith("pixel_values")}
            img_features = F.normalize(model.get_image_features(**img_kwargs), p=2, dim=1)
            fused = (text_features + img_features) / 2
            return F.normalize(fused, p=2, dim=1)
    return text_features


def _candidate_features(model, processor, images: Sequence[Image.Image], device: str) -> torch.Tensor:
    inputs = processor(images=list(images), return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=pixel_values)
    return F.normalize(feats, p=2, dim=1)


def rank_with_clip(
    *,
    manifest: Path,
    model_name: str,
    output_dir: Path,
    top_k: Sequence[int],
    device: str,
    limit: int | None = None,
) -> int:
    model, processor = _load_clip(model_name, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {k: (output_dir / f"top{k}.jsonl").open("w", encoding="utf-8") for k in top_k}

    n = 0
    try:
        for example in tqdm(load_candidate_pool(manifest), desc="clip-rank"):
            if limit is not None and n >= limit:
                break
            question_image = (
                Image.open(example.question_image_path).convert("RGB")
                if example.question_image_path
                else None
            )
            cand_imgs = [Image.open(c.image_path).convert("RGB") for c in example.candidate_images]
            if not cand_imgs:
                continue

            q_feat = _query_features(model, processor, example.query, question_image, device)
            c_feats = _candidate_features(model, processor, cand_imgs, device)
            sims = (q_feat @ c_feats.T).squeeze(0).cpu().tolist()
            scored = [
                {
                    "image_id": c.image_id,
                    "image_path": c.image_path,
                    "score": float(s),
                    "source": c.source,
                }
                for c, s in zip(example.candidate_images, sims)
            ]

            for k in top_k:
                payload = {
                    "qid": example.qid,
                    "query": example.query,
                    "top_k": int(k),
                    "selected_images": select_top_k(scored, int(k)),
                    "answer": example.answer,
                    "choices": example.choices,
                    "metadata": example.metadata,
                    "gt_image_ids": example.gt_image_ids,
                }
                files[k].write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    finally:
        for f in files.values():
            f.close()
    return n


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    n = rank_with_clip(
        manifest=Path(args.manifest),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        device=args.device,
        limit=args.limit,
    )
    print(f"Ranked {n} examples with CLIP -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
