#!/usr/bin/env python3
"""Embed values from a pscc_labels.csv using sentence-transformers and save as a torch file.

Usage example:
    python scripts/embed_pscc_labels.py --model all-MiniLM-L6-v2 --output_file data/generated/pscc_embeddings.pt

The script will try to locate `pscc_labels.csv` in the repository root or current directory
by default, but a custom path can be provided with `--csv`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
import argparse
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer


def find_csv(candidates=None) -> Path:
    if candidates is None:
        here = Path(__file__).resolve().parent
        candidates = [here / ".." / "pscc_labels.csv", here / "pscc_labels.csv", Path("pscc_labels.csv")]
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("pscc_labels.csv not found in candidate locations")


def load_pscc_labels(path: Path) -> Dict[str, List[str]]:
    """Load the pscc labels CSV into a mapping: key -> list of label values.

    The CSV format is expected to have the property name in the first column and
    values in the subsequent columns which may contain pipe-separated values.
    This mirrors the behavior in `pydantic_schema.py`.
    """
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        # fallback to a reasonable limit
        try:
            csv.field_size_limit(10 * 1024 * 1024)
        except Exception:
            pass

    labels: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            key = row[0].strip().lower()
            if not key:
                continue
            value = "|".join(row[1:]) if len(row) > 1 else ""
            items = [item.strip() for item in str(value).split("|") if item.strip()]
            if items:
                # preserve insertion order and allow duplicates if present
                labels.setdefault(key, []).extend(items)
    return labels


def build_value_index(labels: Dict[str, List[str]]):
    """Flatten label values and produce mapping from key -> list of indices into flat values list."""
    values: List[str] = []
    mapping: Dict[str, List[int]] = {}
    for key, items in labels.items():
        idxs = []
        for it in items:
            idx = len(values)
            values.append(it)
            idxs.append(idx)
        mapping[key] = idxs
    return values, mapping


def embed_values(model_name: str, values: List[str], device: str = None, batch_size: int = 64):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    # SentenceTransformer.encode returns numpy array
    embeddings = model.encode(values, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    tensor = torch.from_numpy(embeddings).to(torch.float32)
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Embed pscc_labels.csv values with SentenceTransformers and save torch file")
    parser.add_argument("--model", required=True, help="SentenceTransformers model name (e.g. all-MiniLM-L6-v2)")
    parser.add_argument("--output_file", required=True, help="Path to output .pt file (torch.save will be used)")
    parser.add_argument("--csv", default=None, help="Optional path to pscc_labels.csv")
    parser.add_argument("--device", default=None, help="Device to run model on (cpu or cuda). Defaults to cuda if available, else cpu")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    if csv_path is None:
        csv_path = find_csv()
    else:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading CSV from: {csv_path}")
    labels = load_pscc_labels(csv_path)
    if not labels:
        print("No labels found in CSV.")
        return

    print(f"Found {len(labels)} fields. Flattening values...")
    values, mapping = build_value_index(labels)
    print(f"Total values to embed: {len(values)}")

    # Prepare texts to embed: remove any leading code up to the first ", "
    # (do not modify `values` which will be written to the labels.txt file).
    def _strip_code(s: str) -> str:
        if ", " in s:
            return s.split(", ", 1)[1]
        return s

    values_for_embedding = [_strip_code(v) for v in values]

    print(f"Loading model '{args.model}' on device '{args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}'")
    embeddings = embed_values(args.model, values_for_embedding, device=args.device, batch_size=args.batch_size)

    out = {
        "keys": list(labels.keys()),
        "values": values,
        "mapping": mapping,
        "embeddings": embeddings,
        "meta": {"model": args.model},
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving embeddings to: {out_path}")
    torch.save(out, str(out_path))
    # Also save label names (one per line) next to the .pt file
    labels_path = out_path.with_name(out_path.name + ".labels.txt")
    print(f"Saving label names to: {labels_path}")
    with labels_path.open("w", encoding="utf-8") as lf:
        for v in values:
            lf.write(f"{v}\n")

    print("Done.")


if __name__ == "__main__":
    main()
