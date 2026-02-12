#!/usr/bin/env python3
"""Map entity texts from JSON files to nearest label embeddings and count matches.

The script takes glob patterns for JSON files containing objects like:
  {"entities": [{"entity": "TIME", "word": "2 weeks"}, ...]}

It filters entities by type `TIME` or `POSOLOGY`, embeds the entity `word` values
with a SentenceTransformer model, finds the nearest label embedding from a
provided .pt file, increments a per-label counter, and writes a JSON counts
file. Counts are updated after each processed file so partial results persist.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def find_label_txt(pt_path: Path) -> Path:
    # matches the naming convention used by embed_pscc_labels.py
    candidate = pt_path.with_name(pt_path.name + ".labels.txt")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Labels txt not found at {candidate}")


def load_label_embeddings(pt_path: Path, device: str = "cpu") -> (torch.Tensor, List[str]):
    data = torch.load(str(pt_path), map_location="cpu")
    if isinstance(data, dict) and "embeddings" in data:
        emb = data["embeddings"]
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
    elif isinstance(data, torch.Tensor):
        emb = data
    else:
        raise ValueError("Unsupported .pt format: expected dict with 'embeddings' or a tensor")

    labels_txt = find_label_txt(pt_path)
    with labels_txt.open("r", encoding="utf-8") as fh:
        labels = [line.rstrip("\n") for line in fh]

    if emb.shape[0] != len(labels):
        # allow shorter/longer but warn
        print(f"Warning: {pt_path} embeddings rows={emb.shape[0]} != labels lines={len(labels)}")
        n = min(emb.shape[0], len(labels))
        emb = emb[:n]
        labels = labels[:n]

    emb = emb.to(device)
    emb = F.normalize(emb, dim=1)
    return emb, labels


def expand_globs(patterns: List[str]) -> List[Path]:
    files = []
    for p in patterns:
        files.extend([Path(x) for x in glob.glob(p, recursive=True)])
    # unique
    uniq = list(dict.fromkeys(files))
    return uniq


def save_counts_json(out_path: Path, counts: Dict[str, int]):
    # `counts` is a mapping label -> int. Sort by count descending and write
    # JSON without key sorting to preserve the order.
    pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    mapping = {name: cnt for name, cnt in pairs}
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False, sort_keys=False)
    tmp.replace(out_path)


def main():
    parser = argparse.ArgumentParser(description="Map entities to label embeddings and count matches")
    parser.add_argument("patterns", nargs='+', help="Glob patterns to JSON files to process")
    parser.add_argument("--labels_pt", required=True, help=".pt file containing label embeddings")
    parser.add_argument("--model", required=True, help="SentenceTransformers model name for embedding entities")
    parser.add_argument("--output", required=True, help="JSON output file path for counts")
    parser.add_argument("--device", default=None, help="Device to run on (cpu or cuda). Defaults to cuda if available")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--topk", type=int, default=9, help="Number of unique top label names to count per entity")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    labels_pt = Path(args.labels_pt)
    if not labels_pt.exists():
        raise FileNotFoundError(f"Labels .pt file not found: {labels_pt}")

    print(f"Loading label embeddings from: {labels_pt}")
    label_embs, label_names = load_label_embeddings(labels_pt, device=device)
    n_labels = label_embs.shape[0]
    # Use a mapping from label name -> count so we don't need to index by position.
    counts: Dict[str, int] = {name: 0 for name in label_names}

    # Expand and shuffle files
    files = expand_globs(args.patterns)
    if not files:
        print("No files matched the provided patterns.")
        return
    random.shuffle(files)

    model = SentenceTransformer(args.model, device=device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Iterate with tqdm showing remaining files
    for fp in tqdm(files, desc="Processing files"):
        try:
            with fp.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as e:
            print(f"Failed to load {fp}: {e}")
            # update partial results and continue
            save_counts_json(out_path, counts)
            continue

        ents = doc.get("entities") or []
        texts = [e.get("word") for e in ents if e.get("entity") in ("TIME", "POSOLOGY") and e.get("word")]
        if not texts:
            # still write partial results
            save_counts_json(out_path, counts)
            continue

        # embed texts in batches via model.encode
        emb = model.encode(texts, convert_to_tensor=True, batch_size=args.batch_size, show_progress_bar=False)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = emb.to(device)
        emb = F.normalize(emb, dim=1)

        # compute cosine similarity and pick top candidates
        sims = torch.matmul(emb.to(label_embs.dtype), label_embs.t())
        # For each entity: search top (3 * K_SELECT) indices, then select up to
        # K_SELECT unique label NAMES (this avoids incrementing the same label
        # name multiple times when duplicates exist).
        K_SELECT = args.topk
        K_SEARCH = 3 * K_SELECT
        k_search = min(K_SEARCH, sims.shape[1])
        vals, idxs = torch.topk(sims, k=k_search, dim=1)
        # idxs: (n_texts, k_search)
        for row in idxs.cpu().tolist():
            top_label_names = set()
            for idx in row:
                name = label_names[int(idx)]
                if name in top_label_names:
                    continue
                counts[name] = counts.get(name, 0) + 1
                top_label_names.add(name)
                if len(top_label_names) >= K_SELECT:
                    break

        # write partial counts after each file
        save_counts_json(out_path, counts)

    # final save (already saved incrementally but ensure final)
    save_counts_json(out_path, counts)
    print(f"Finished. Wrote counts to {out_path}")


if __name__ == "__main__":
    main()
