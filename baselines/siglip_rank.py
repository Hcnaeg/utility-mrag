"""SigLIP-based candidate ranking.

Mirrors :mod:`baselines.clip_rank` but uses ``transformers``' SigLIP API.

Usage::

    uv run python -m baselines.siglip_rank \\
        --manifest data/manifests/mrag_bench_candidates.jsonl \\
        --model google/siglip-base-patch16-384 \\
        --top_k 1 3 5 \\
        --output_dir outputs/baselines/siglip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utility_mrag.data.candidate_pool import load_candidate_pool
from utility_mrag.selection.topk import select_top_k


def _load_siglip(model_name: str, device: str):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def _text_features(model, processor, text: str, device: str) -> torch.Tensor:
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return F.normalize(feats, p=2, dim=1)


def _image_features(model, processor, images: Sequence[Image.Image], device: str) -> torch.Tensor:
    inputs = processor(images=list(images), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=pixel_values)
    return F.normalize(feats, p=2, dim=1)


def rank_with_siglip(
    *,
    manifest: Path,
    model_name: str,
    output_dir: Path,
    top_k: Sequence[int],
    device: str,
    limit: int | None = None,
) -> int:
    model, processor = _load_siglip(model_name, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {k: (output_dir / f"top{k}.jsonl").open("w", encoding="utf-8") for k in top_k}
    n = 0
    try:
        for example in tqdm(load_candidate_pool(manifest), desc="siglip-rank"):
            if limit is not None and n >= limit:
                break
            cand_imgs = [Image.open(c.image_path).convert("RGB") for c in example.candidate_images]
            if not cand_imgs:
                continue
            q_feat = _text_features(model, processor, example.query, device)
            c_feats = _image_features(model, processor, cand_imgs, device)
            sims = (q_feat @ c_feats.T).squeeze(0).cpu().tolist()
            scored = [
                {"image_id": c.image_id, "image_path": c.image_path, "score": float(s), "source": c.source}
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
    parser.add_argument("--model", default="google/siglip-base-patch16-384")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    n = rank_with_siglip(
        manifest=Path(args.manifest),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        device=args.device,
        limit=args.limit,
    )
    print(f"Ranked {n} examples with SigLIP -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
