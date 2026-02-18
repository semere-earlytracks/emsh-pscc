#!/usr/bin/env python3
"""Embed text values from CSV files in pssc-labels folder using sentence-transformers.

This script reads all CSV files in the pssc-labels directory, embeds the 'text' column
using a specified SentenceTransformers model, and saves the embeddings as .pt files
alongside the original CSV files.

Usage example:
    python scripts/embed_pssc_labels_v2.py --model all-MiniLM-L6-v2 --labels_dir data/pssc-labels
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict

import torch
from sentence_transformers import SentenceTransformer


def load_csv_data(csv_path: Path) -> Dict[str, List[str]]:
    """Load CSV file and return labels and text values.
    
    Returns:
        Dictionary with 'labels' and 'texts' keys containing lists of strings.
    """
    labels = []
    texts = []
    
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row.get("label", "").strip()
            text = row.get("text", "").strip()
            if label and text:
                labels.append(label)
                texts.append(text)
    
    return {"labels": labels, "texts": texts}


def embed_texts(
    model_name: str,
    texts: List[str],
    device: str = None,
    batch_size: int = 64
) -> torch.Tensor:
    """Embed a list of texts using SentenceTransformer.
    
    Args:
        model_name: Name of the SentenceTransformer model
        texts: List of text strings to embed
        device: Device to use (cpu or cuda)
        batch_size: Batch size for encoding
        
    Returns:
        Tensor of shape (len(texts), embedding_dim) with normalized embeddings
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    
    # Encode texts
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Convert to torch tensor
    tensor = torch.from_numpy(embeddings).to(torch.float32)
    return tensor


def save_embeddings(
    output_path: Path,
    labels: List[str],
    texts: List[str],
    embeddings: torch.Tensor,
    model_name: str
):
    """Save embeddings and metadata to a .pt file.
    
    Args:
        output_path: Path to save the .pt file
        labels: List of label strings
        texts: List of text strings
        embeddings: Tensor of embeddings
        model_name: Name of the model used
    """
    data = {
        "labels": labels,
        "texts": texts,
        "embeddings": embeddings,
        "meta": {"model": model_name}
    }
    
    torch.save(data, str(output_path))
    print(f"  Saved embeddings to: {output_path}")


def process_csv_file(
    csv_path: Path,
    model_name: str,
    device: str,
    batch_size: int
):
    """Process a single CSV file and create embeddings.
    
    Args:
        csv_path: Path to the CSV file
        model_name: SentenceTransformer model name
        device: Device to use
        batch_size: Batch size for encoding
    """
    print(f"\nProcessing: {csv_path.name}")
    
    # Load data
    data = load_csv_data(csv_path)
    labels = data["labels"]
    texts = data["texts"]
    
    if not texts:
        print(f"  Warning: No data found in {csv_path.name}")
        return
    
    print(f"  Found {len(texts)} entries")
    
    # Create embeddings
    embeddings = embed_texts(model_name, texts, device, batch_size)
    
    # Save with same name but .pt extension
    output_path = csv_path.with_suffix(".pt")
    save_embeddings(output_path, labels, texts, embeddings, model_name)


def main():
    parser = argparse.ArgumentParser(
        description="Embed text values from pssc-labels CSV files"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="SentenceTransformers model name (e.g., all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--labels_dir",
        default="data/pssc-labels",
        help="Directory containing CSV files (default: data/pssc-labels)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (cpu or cuda). Auto-detects if not specified"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64)"
    )
    args = parser.parse_args()
    
    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Find all CSV files
    csv_files = sorted(labels_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {labels_dir}")
        return
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using model: {args.model}")
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_path in csv_files:
        try:
            process_csv_file(csv_path, args.model, device, args.batch_size)
        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")
            continue
    
    print("\nDone! All embeddings created.")


if __name__ == "__main__":
    main()
