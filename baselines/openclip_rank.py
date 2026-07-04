"""OpenCLIP-based candidate ranking.

Requires the ``open-clip-torch`` extra (``uv sync --extra clip``).

Usage::

    uv run python -m baselines.openclip_rank \\
        --manifest data/manifests/mrag_bench_candidates.jsonl \\
        --model ViT-L-14 --pretrained openai \\
        --top_k 1 3 5 \\
        --output_dir outputs/baselines/openclip
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


def _load_openclip(model_name: str, pretrained: str, device: str):
    try:
        import open_clip  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "open_clip is required. Install with `uv sync --extra clip`."
        ) from exc
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def rank_with_openclip(
    *,
    manifest: Path,
    model_name: str,
    pretrained: str,
    output_dir: Path,
    top_k: Sequence[int],
    device: str,
    limit: int | None = None,
) -> int:
    model, preprocess, tokenizer = _load_openclip(model_name, pretrained, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {k: (output_dir / f"top{k}.jsonl").open("w", encoding="utf-8") for k in top_k}
    n = 0
    try:
        for example in tqdm(load_candidate_pool(manifest), desc="openclip-rank"):
            if limit is not None and n >= limit:
                break
            cand_imgs = [
                preprocess(Image.open(c.image_path).convert("RGB")).unsqueeze(0)
                for c in example.candidate_images
            ]
            if not cand_imgs:
                continue
            cand_tensor = torch.cat(cand_imgs, dim=0).to(device)
            text_tokens = tokenizer([example.query]).to(device)
            with torch.no_grad():
                txt_feats = F.normalize(model.encode_text(text_tokens), p=2, dim=1)
                img_feats = F.normalize(model.encode_image(cand_tensor), p=2, dim=1)
            sims = (txt_feats @ img_feats.T).squeeze(0).cpu().tolist()
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
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    n = rank_with_openclip(
        manifest=Path(args.manifest),
        model_name=args.model,
        pretrained=args.pretrained,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        device=args.device,
        limit=args.limit,
    )
    print(f"Ranked {n} examples with OpenCLIP -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
