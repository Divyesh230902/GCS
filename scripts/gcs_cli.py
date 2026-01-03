#!/usr/bin/env python3
"""
Low-code CLI for team testing:
1) Index arbitrary "data dump" folders
2) Build embeddings + enhanced knowledge graph artifacts
3) Run text/image (multimodal) queries against saved artifacts

This intentionally wraps existing core logic without changing it.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root + src are importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.clip_embeddings import CLIPEmbeddingExtractor
from src.data_dump import DataDumpItem, index_data_dump
from src.enhanced_graphrag import EnhancedGraphRAGRetriever
from src.graphRAG import MedicalKnowledgeGraph
from src.ssm import SSMQueryProcessor


def _parse_level(value: str) -> Optional[int]:
    value = value.strip().lower()
    if value in {"none", "null", "off"}:
        return None
    return int(value)


def _sanitize_cache_key(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "dataset"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec) + 1e-12)
    return (vec / denom).astype(np.float32, copy=False)


def _fuse_embeddings(text_emb: Optional[np.ndarray], image_emb: Optional[np.ndarray]) -> np.ndarray:
    if text_emb is None and image_emb is None:
        raise ValueError("Provide at least one of text or image query.")
    if text_emb is None:
        return _l2_normalize(image_emb)
    if image_emb is None:
        return _l2_normalize(text_emb)
    return _l2_normalize((_l2_normalize(text_emb) + _l2_normalize(image_emb)) / 2.0)


def _build_embeddings(
    items: List[DataDumpItem],
    *,
    clip: CLIPEmbeddingExtractor,
    batch_size: int,
    use_cache: bool,
) -> List[Any]:
    by_dataset: Dict[str, List[DataDumpItem]] = {}
    for item in items:
        by_dataset.setdefault(item.dataset, []).append(item)

    all_embeddings: List[Any] = []
    for dataset, group in sorted(by_dataset.items(), key=lambda x: x[0]):
        group = sorted(group, key=lambda x: x.image_path)
        image_paths = [g.image_path for g in group]
        class_labels = [(g.class_label or "unlabeled") for g in group]

        cache_key = _sanitize_cache_key(dataset)
        embeddings = clip.batch_extract_embeddings(
            image_paths=image_paths,
            class_labels=class_labels,
            dataset=cache_key,
            batch_size=batch_size,
            save_cache=use_cache,
        )

        # Restore original dataset + attach per-item metadata (rel_path/root/etc).
        meta_by_path = {g.image_path: g for g in group}
        for emb in embeddings:
            emb.dataset = dataset
            item = meta_by_path.get(emb.image_path)
            if item is not None:
                emb.metadata = dict(item.metadata)
                emb.metadata["rel_path"] = item.rel_path
        all_embeddings.extend(embeddings)

    return all_embeddings


def cmd_build(args: argparse.Namespace) -> int:
    data_roots = args.data_root
    artifact_path = Path(args.artifact).expanduser()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    items = index_data_dump(
        data_roots,
        dataset_level=args.dataset_level,
        label_level=_parse_level(args.label_level),
        label_default=args.label_default,
        follow_symlinks=args.follow_symlinks,
        max_items=args.max_images,
    )
    if not items:
        print("No images found. Check --data-root and supported extensions.")
        return 2

    print(f"Indexed {len(items)} images from: {', '.join(data_roots)}")

    clip = CLIPEmbeddingExtractor(
        model_name=args.clip_model,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    ssm = SSMQueryProcessor()
    graph = MedicalKnowledgeGraph()
    retriever = EnhancedGraphRAGRetriever(clip_extractor=clip, ssm_processor=ssm, graph=graph)

    image_embeddings = _build_embeddings(items, clip=clip, batch_size=args.batch_size, use_cache=not args.no_cache)
    retriever.build_enhanced_graph(image_embeddings)

    extra = {
        "data_roots": data_roots,
        "dataset_level": args.dataset_level,
        "label_level": args.label_level,
        "label_default": args.label_default,
        "indexed_images": len(items),
        "embedded_images": len(image_embeddings),
    }
    retriever.save(str(artifact_path), extra=extra)
    print(f"Saved artifacts to: {artifact_path}")
    return 0


def _print_results(result, top_k: int) -> None:
    print("\nTop results:")
    for i, img in enumerate(result.retrieved_images[:top_k], 1):
        sim = img.metadata.get("similarity")
        sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) and not math.isnan(sim) else "n/a"
        print(f"{i:>2}. sim={sim_str} dataset={img.dataset} label={img.class_label} path={img.image_path}")


def cmd_query(args: argparse.Namespace) -> int:
    clip = CLIPEmbeddingExtractor(
        model_name=args.clip_model,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    ssm = SSMQueryProcessor()
    retriever, extra = EnhancedGraphRAGRetriever.load(
        args.artifact,
        clip_extractor=clip,
        ssm_processor=ssm,
        graph=MedicalKnowledgeGraph(),
    )

    text_emb = clip.extract_text_embedding(args.text) if args.text else None
    image_emb = None
    if args.image:
        q = clip.extract_image_embedding(args.image, class_label="query", dataset="query", metadata={"query": True})
        image_emb = q.embedding

    query_embedding = _fuse_embeddings(text_emb, image_emb)
    query_desc = args.text or (f"image:{args.image}" if args.image else "")
    result = retriever.retrieve(query=query_desc, top_k=args.top_k, search_mode=args.mode, query_embedding=query_embedding)
    _print_results(result, top_k=args.top_k)
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    clip = CLIPEmbeddingExtractor(
        model_name=args.clip_model,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    ssm = SSMQueryProcessor()
    retriever, extra = EnhancedGraphRAGRetriever.load(
        args.artifact,
        clip_extractor=clip,
        ssm_processor=ssm,
        graph=MedicalKnowledgeGraph(),
    )

    # Find the node_id for this image if it exists in the artifact.
    target_node_id = None
    for node_id, meta in retriever.node_metadata.items():
        if meta.get("image_path") == args.image:
            target_node_id = node_id
            break

    if target_node_id is None:
        q = clip.extract_image_embedding(args.image, class_label="query", dataset="query", metadata={"query": True})
        query_embedding = _l2_normalize(q.embedding)
    else:
        query_embedding = _l2_normalize(retriever.node_embeddings[target_node_id])

    # Rank nodes by similarity and tag top-k.
    sims: List[Tuple[str, float]] = []
    for node_id, emb in retriever.node_embeddings.items():
        if not node_id.startswith("img_"):
            continue
        sim = float(np.dot(query_embedding, _l2_normalize(emb)))
        sims.append((node_id, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [nid for nid, _ in sims[: args.k]]

    for nid in top_nodes:
        meta = retriever.node_metadata.get(nid, {})
        tags = meta.get("user_tags")
        if not isinstance(tags, list):
            tags = []
        if args.tag not in tags:
            tags.append(args.tag)
        meta["user_tags"] = tags
        retriever.node_metadata[nid] = meta

    out_path = args.out or args.artifact
    retriever.save(out_path, extra=extra)
    print(f"Tagged {len(top_nodes)} images with '{args.tag}'. Saved: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gcs_cli.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
        sp.add_argument("--device", default="auto")
        sp.add_argument("--cache-dir", default="./embeddings_cache")

    b = sub.add_parser("build", help="Index data-dump roots and save artifacts")
    add_common(b)
    b.add_argument("--data-root", action="append", required=True, help="Repeatable; folders to scan recursively")
    b.add_argument("--artifact", default="./artifacts/gcs_artifacts.pkl")
    b.add_argument("--dataset-level", type=int, default=0, help="Relative path segment index; -1 uses root name")
    b.add_argument("--label-level", default="1", help="Relative path segment index, or 'none'")
    b.add_argument("--label-default", default="unlabeled")
    b.add_argument("--follow-symlinks", action="store_true")
    b.add_argument("--max-images", type=int, default=None)
    b.add_argument("--batch-size", type=int, default=32)
    b.add_argument("--no-cache", action="store_true")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="Run a text/image query against saved artifacts")
    add_common(q)
    q.add_argument("--artifact", default="./artifacts/gcs_artifacts.pkl")
    q.add_argument("--text", default=None)
    q.add_argument("--image", default=None)
    q.add_argument("--mode", default="auto", choices=["auto", "global", "local", "hybrid"])
    q.add_argument("--top-k", type=int, default=10)
    q.set_defaults(func=cmd_query)

    t = sub.add_parser("tag", help="Single-shot tagging propagated in embedding space")
    add_common(t)
    t.add_argument("--artifact", default="./artifacts/gcs_artifacts.pkl")
    t.add_argument("--image", required=True, help="Seed image path (must exist on disk)")
    t.add_argument("--tag", required=True)
    t.add_argument("--k", type=int, default=50, help="How many nearest images to tag")
    t.add_argument("--out", default=None, help="Optional output artifact path (defaults to overwrite)")
    t.set_defaults(func=cmd_tag)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

