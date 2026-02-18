#!/usr/bin/env python3
"""Process JSON files and replace field values with top-1 label suggestions from embeddings.

This script reads all JSON files from an input directory (recursively), replaces certain fields
with their top-1 matched labels from pre-computed embeddings, and writes the modified JSON files
to an output directory with the same structure.

Usage example:
    python scripts/infer_pssc_labelV2.py \
        --model all-MiniLM-L6-v2 \
        --input_dir data/raw_annotations \
        --output_dir data/processed_annotations \
        --labels_dir data/pssc-labels \
        --batch_size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Mapping of field names to their corresponding embedding types
FIELD_TO_EMBEDDING_TYPE = {
    "relatedpathologycode": "relatedpathologycode_ext",
    "topographycode": "topographycode_ext",
    "metastasistopocode": "topographycode_ext",
    "specimentopographycode": "topographycode_ext",
    "morphologycode": "morphologycode_ext",
    "surgerytype": "surgerytype_ext",
    "moleculecode": "moleculescode_ext",
    "biomarkername": "biomarkername_ext",
}


class EmbeddingCache:
    """Cache for pre-computed embeddings."""
    
    def __init__(self, labels_dir: Path, device: str = "cpu"):
        self.labels_dir = labels_dir
        self.device = device
        self.cache: Dict[str, Dict] = {}
    
    def load(self, embedding_type: str) -> Dict:
        """Load embeddings for a given type (cached)."""
        if embedding_type not in self.cache:
            embedding_path = self.labels_dir / f"{embedding_type}.pt"
            if not embedding_path.exists():
                raise FileNotFoundError(
                    f"Embedding file not found: {embedding_path}\n"
                    f"Please run embed_pssc_labels_v2.py first."
                )
            
            data = torch.load(str(embedding_path), map_location=self.device)
            
            # Ensure embeddings are normalized
            embeddings = data["embeddings"]
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)
            
            embeddings = embeddings.to(self.device)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            data["embeddings"] = embeddings
            
            self.cache[embedding_type] = data
        
        return self.cache[embedding_type]


class BatchInferencer:
    """Batch inference engine for mapping texts to labels."""
    
    def __init__(
        self,
        model_name: str,
        embedding_cache: EmbeddingCache,
        device: str = "cpu",
        batch_size: int = 64
    ):
        self.model_name = model_name
        self.embedding_cache = embedding_cache
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
    
    def infer_batch(
        self,
        texts: List[str],
        embedding_type: str
    ) -> List[Optional[str]]:
        """Infer top-1 labels for a batch of texts.
        
        Args:
            texts: List of text strings to map to labels
            embedding_type: Type of embedding to use
            
        Returns:
            List of top-1 label strings (or None for empty texts)
        """
        if not texts:
            return []
        
        # Filter out None/empty texts and track their positions
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and str(text).strip():
                valid_texts.append(str(text).strip())
                valid_indices.append(i)
        
        # Initialize results with None
        results = [None] * len(texts)
        
        if not valid_texts:
            return results
        
        # Load embeddings for this type
        embedding_data = self.embedding_cache.load(embedding_type)
        label_embeddings = embedding_data["embeddings"]
        labels = embedding_data["labels"]
        
        # Embed query texts
        query_embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        if not isinstance(query_embeddings, torch.Tensor):
            query_embeddings = torch.tensor(query_embeddings)
        
        query_embeddings = query_embeddings.to(self.device)
        
        # Compute similarities and get top-1
        similarities = torch.matmul(
            query_embeddings.to(label_embeddings.dtype),
            label_embeddings.t()
        )
        
        top_indices = torch.argmax(similarities, dim=1)
        
        # Map back to original positions
        for i, idx in enumerate(top_indices.cpu().tolist()):
            original_pos = valid_indices[i]
            results[original_pos] = labels[idx]
        
        return results


def collect_texts_from_json(data: Any, field_names: set) -> Dict[str, List[tuple]]:
    """Recursively collect text values for specified fields from JSON data.
    
    Args:
        data: JSON data (dict, list, or primitive)
        field_names: Set of field names to collect
        
    Returns:
        Dictionary mapping field names to lists of (path, value) tuples
        where path is a list of keys/indices to reach that value
    """
    collected = defaultdict(list)
    
    def recurse(obj, path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = path + [key]
                if key in field_names:
                    collected[key].append((new_path, value))
                recurse(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(item, path + [i])
    
    recurse(data, [])
    return collected


def set_value_by_path(data: Any, path: List, value: Any):
    """Set a value in nested data structure using a path."""
    if not path:
        return value
    
    current = data
    for i, key in enumerate(path[:-1]):
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, list):
            current = current[key]
    
    last_key = path[-1]
    if isinstance(current, dict):
        current[last_key] = value
    elif isinstance(current, list):
        current[last_key] = value


def process_json_file(
    input_path: Path,
    output_path: Path,
    inferencer: BatchInferencer,
    field_names: set,
    field_to_type: Dict[str, str]
):
    """Process a single JSON file and write the result.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        inferencer: BatchInferencer instance
        field_names: Set of field names to process
        field_to_type: Mapping of field names to embedding types
    """
    # Load JSON
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    
    # Collect all texts for each field
    collected = collect_texts_from_json(data, field_names)
    
    # For each field, batch infer and replace
    for field_name, paths_and_values in collected.items():
        if not paths_and_values:
            continue
        
        embedding_type = field_to_type[field_name]
        texts = [value for _, value in paths_and_values]
        
        # Get top-1 predictions
        predictions = inferencer.infer_batch(texts, embedding_type)
        
        # Replace values in data
        for (path, original_value), predicted_label in zip(paths_and_values, predictions):
            if predicted_label is not None:
                set_value_by_path(data, path, predicted_label)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def find_all_json_files(input_dir: Path) -> List[Path]:
    """Find all JSON files recursively in a directory."""
    return sorted(input_dir.rglob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON files and replace field values with label predictions"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="SentenceTransformers model name (must match embedding model)"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing JSON files"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for processed JSON files"
    )
    parser.add_argument(
        "--labels_dir",
        default="data/pssc-labels",
        help="Directory containing embedding .pt files (default: data/pssc-labels)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding queries (default: 64)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cpu or cuda). Auto-detects if not specified"
    )
    args = parser.parse_args()
    
    # Setup
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    labels_dir = Path(args.labels_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Find all JSON files
    json_files = find_all_json_files(input_dir)
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Using model: {args.model}")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize components
    embedding_cache = EmbeddingCache(labels_dir, device=device)
    inferencer = BatchInferencer(
        args.model,
        embedding_cache,
        device=device,
        batch_size=args.batch_size
    )
    
    field_names = set(FIELD_TO_EMBEDDING_TYPE.keys())
    
    # Process each file
    for json_path in tqdm(json_files, desc="Processing JSON files"):
        try:
            # Compute relative path to maintain directory structure
            rel_path = json_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            process_json_file(
                json_path,
                output_path,
                inferencer,
                field_names,
                FIELD_TO_EMBEDDING_TYPE
            )
        except Exception as e:
            print(f"\nError processing {json_path}: {e}")
            continue
    
    print(f"\nDone! Processed files written to {output_dir}")


if __name__ == "__main__":
    main()
