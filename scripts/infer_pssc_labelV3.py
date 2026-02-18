#!/usr/bin/env python3
"""Process JSON files in two passes: first collect and map all unique strings, then replace.

This script operates in two passes:
1. Pass 1: Collect all unique strings from specified fields across all JSON files,
   batch process them to get label mappings, and save mappings to disk.
2. Pass 2: Read JSON files again and replace field values using the saved mappings.

This approach is more efficient as each unique string is only processed once.

Usage example:
    python scripts/infer_pssc_labelV3.py \
        --model all-MiniLM-L6-v2 \
        --input_dir data/raw_annotations \
        --output_dir data/processed_annotations \
        --labels_dir data/pssc-labels \
        --cache_dir data/label_cache \
        --batch_size 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
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
        batch_size: int = 128
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
    ) -> List[str]:
        """Infer top-1 labels for a batch of texts.
        
        Args:
            texts: List of text strings to map to labels (must be non-empty)
            embedding_type: Type of embedding to use
            
        Returns:
            List of top-1 label strings
        """
        if not texts:
            return []
        
        # Load embeddings for this type
        embedding_data = self.embedding_cache.load(embedding_type)
        label_embeddings = embedding_data["embeddings"]
        labels = embedding_data["labels"]
        
        # Embed query texts
        query_embeddings = self.model.encode(
            texts,
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
        
        # Get labels
        results = [labels[idx] for idx in top_indices.cpu().tolist()]
        
        return results


def collect_field_values_from_json(data: Any, field_names: Set[str]) -> Dict[str, Set[str]]:
    """Recursively collect unique string values for specified fields from JSON data.
    
    Args:
        data: JSON data (dict, list, or primitive)
        field_names: Set of field names to collect
        
    Returns:
        Dictionary mapping field names to sets of unique non-empty string values
    """
    collected = defaultdict(set)
    
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in field_names:
                    # Collect the value if it's a non-empty string
                    if value and isinstance(value, str) and value.strip():
                        collected[key].add(value.strip())
                recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(data)
    return collected


def collect_all_strings_pass1(
    input_dir: Path,
    field_names: Set[str]
) -> Dict[str, Set[str]]:
    """Pass 1: Collect all unique strings from all JSON files for each field type.
    
    Args:
        input_dir: Directory containing JSON files
        field_names: Set of field names to collect
        
    Returns:
        Dictionary mapping field names to sets of unique strings
    """
    json_files = sorted(input_dir.rglob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return {}
    
    print(f"Pass 1: Collecting strings from {len(json_files)} JSON files...")
    
    all_collected = defaultdict(set)
    
    for json_path in tqdm(json_files, desc="Collecting strings"):
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            collected = collect_field_values_from_json(data, field_names)
            
            # Merge into all_collected
            for field_name, values in collected.items():
                all_collected[field_name].update(values)
        
        except Exception as e:
            print(f"\nError reading {json_path}: {e}")
            continue
    
    return all_collected


def create_string_to_label_mappings(
    collected_strings: Dict[str, Set[str]],
    field_to_type: Dict[str, str],
    inferencer: BatchInferencer,
    cache_dir: Path
):
    """Create and save string-to-label mappings for each field type.
    
    Args:
        collected_strings: Dictionary of field names to sets of unique strings
        field_to_type: Mapping of field names to embedding types
        inferencer: BatchInferencer instance
        cache_dir: Directory to save mapping files
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Group fields by embedding type
    type_to_strings = defaultdict(set)
    for field_name, strings in collected_strings.items():
        embedding_type = field_to_type[field_name]
        type_to_strings[embedding_type].update(strings)
    
    print("\nCreating string-to-label mappings...")
    
    # Process each embedding type
    for embedding_type, strings in type_to_strings.items():
        if not strings:
            continue
        
        print(f"\nProcessing {embedding_type}: {len(strings)} unique strings")
        
        # Convert set to sorted list for consistency
        string_list = sorted(strings)
        
        # Batch process all strings
        labels = inferencer.infer_batch(string_list, embedding_type)
        
        # Create mapping
        mapping = {string: label for string, label in zip(string_list, labels)}
        
        # Save to disk
        cache_path = cache_dir / f"{embedding_type}_mapping.json"
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2, ensure_ascii=False)
        
        print(f"  Saved mapping to: {cache_path}")


def load_mappings(cache_dir: Path, field_to_type: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Load all string-to-label mappings from cache directory.
    
    Args:
        cache_dir: Directory containing mapping files
        field_to_type: Mapping of field names to embedding types
        
    Returns:
        Dictionary mapping field names to their string-to-label dictionaries
    """
    # Get unique embedding types
    embedding_types = set(field_to_type.values())
    
    # Load mappings for each type
    type_mappings = {}
    for embedding_type in embedding_types:
        cache_path = cache_dir / f"{embedding_type}_mapping.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {cache_path}")
        
        with cache_path.open("r", encoding="utf-8") as fh:
            type_mappings[embedding_type] = json.load(fh)
    
    # Create field-specific mappings
    field_mappings = {}
    for field_name, embedding_type in field_to_type.items():
        field_mappings[field_name] = type_mappings[embedding_type]
    
    return field_mappings


def replace_values_in_json(
    data: Any,
    field_names: Set[str],
    field_mappings: Dict[str, Dict[str, str]]
):
    """Recursively replace field values in JSON data using mappings.
    
    Args:
        data: JSON data (dict, list, or primitive) - modified in place
        field_names: Set of field names to replace
        field_mappings: Dictionary mapping field names to string-to-label mappings
    """
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if key in field_names and isinstance(value, str):
                    # Replace with mapped label if available
                    stripped_value = value.strip()
                    if stripped_value in field_mappings[key]:
                        obj[key] = field_mappings[key][stripped_value]
                else:
                    recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(data)


def process_all_files_pass2(
    input_dir: Path,
    output_dir: Path,
    field_names: Set[str],
    field_mappings: Dict[str, Dict[str, str]]
):
    """Pass 2: Process all JSON files and replace field values using mappings.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory for processed files
        field_names: Set of field names to replace
        field_mappings: Dictionary mapping field names to string-to-label mappings
    """
    json_files = sorted(input_dir.rglob("*.json"))
    
    print(f"\nPass 2: Processing {len(json_files)} JSON files...")
    
    for json_path in tqdm(json_files, desc="Replacing values"):
        try:
            # Load JSON
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            # Replace values
            replace_values_in_json(data, field_names, field_mappings)
            
            # Compute output path (preserve directory structure)
            rel_path = json_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Write output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"\nError processing {json_path}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON files in two passes: collect strings, then replace with labels"
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
        "--cache_dir",
        default="data/label_cache",
        help="Directory to store string-to-label mappings (default: data/label_cache)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding queries (default: 128)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cpu or cuda). Auto-detects if not specified"
    )
    parser.add_argument(
        "--skip_pass1",
        action="store_true",
        help="Skip Pass 1 (use existing mappings in cache_dir)"
    )
    args = parser.parse_args()
    
    # Setup
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    labels_dir = Path(args.labels_dir)
    cache_dir = Path(args.cache_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    print(f"Using model: {args.model}")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    field_names = set(FIELD_TO_EMBEDDING_TYPE.keys())
    
    # Pass 1: Collect and map strings
    if not args.skip_pass1:
        collected_strings = collect_all_strings_pass1(input_dir, field_names)
        
        # Print statistics
        print("\nCollected strings by field:")
        for field_name in sorted(collected_strings.keys()):
            print(f"  {field_name}: {len(collected_strings[field_name])} unique values")
        
        # Initialize components for inference
        embedding_cache = EmbeddingCache(labels_dir, device=device)
        inferencer = BatchInferencer(
            args.model,
            embedding_cache,
            device=device,
            batch_size=args.batch_size
        )
        
        # Create and save mappings
        create_string_to_label_mappings(
            collected_strings,
            FIELD_TO_EMBEDDING_TYPE,
            inferencer,
            cache_dir
        )
    else:
        print("Skipping Pass 1 (using existing mappings)")
    
    # Pass 2: Load mappings and process files
    field_mappings = load_mappings(cache_dir, FIELD_TO_EMBEDDING_TYPE)
    
    print(f"\nLoaded mappings from: {cache_dir}")
    for field_name in sorted(field_mappings.keys()):
        print(f"  {field_name}: {len(field_mappings[field_name])} mappings")
    
    process_all_files_pass2(
        input_dir,
        output_dir,
        field_names,
        field_mappings
    )
    
    print(f"\nDone! Processed files written to {output_dir}")


if __name__ == "__main__":
    main()


'''
# Full run (both passes)
python scripts/infer_pssc_labelV3.py \
    --model all-MiniLM-L6-v2 \
    --input_dir data/raw_annotations \
    --output_dir data/processed_annotations \
    --labels_dir data/pssc-labels \
    --cache_dir data/label_cache \
    --batch_size 128

# Skip Pass 1 (use existing mappings)
python scripts/infer_pssc_labelV3.py \
    --model all-MiniLM-L6-v2 \
    --input_dir data/raw_annotations \
    --output_dir data/processed_annotations \
    --cache_dir data/label_cache \
    --skip_pass1
'''