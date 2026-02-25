#!/usr/bin/env python3
"""Postprocessor to remove entries where specified fields have value "other".

This script processes JSON files and removes entire objects/entries when any of the
specified fields (same as infer_pssc_labelV3.py) have the value "other".

For example, if moleculecode is "other", the entire medication entry is removed.

Usage example:
    python scripts/postprocess_remove_other.py \
        --input_dir data/processed_annotations \
        --output_dir data/final_annotations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Set
from collections import Counter

from tqdm import tqdm


# Same fields as in infer_pssc_labelV3.py
FIELD_NAMES_TO_CHECK = {
    "relatedpathologycode",
    "topographycode",
    "metastasistopocode",
    "specimentopographycode",
    "morphologycode",
    "surgerytype",
    "moleculecode",
    "radiotherapytype",
    "imagingmodality",
}


def has_other_value(obj: Any, field_names: Set[str]) -> bool:
    """Check if object contains any field with value "other".
    
    Args:
        obj: Object to check (dict, list, or primitive)
        field_names: Set of field names to check
        
    Returns:
        True if object contains any field with value "other"
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Check if this field has value "other"
            if key in field_names and isinstance(value, str) and value.strip().lower() == "other":
                return True
    return False


def remove_other_entries(data: Any, field_names: Set[str]) -> tuple[Any, int]:
    """Recursively remove entries where specified fields have value "other".
    
    Args:
        data: JSON data (dict, list, or primitive)
        field_names: Set of field names to check
        
    Returns:
        Tuple of (cleaned data, count of removed entries)
    """
    removed_count = 0
    
    def recurse(obj: Any) -> Any:
        nonlocal removed_count
        
        if isinstance(obj, dict):
            # Process each value recursively
            new_dict = {}
            for key, value in obj.items():
                cleaned_value = recurse(value)
                new_dict[key] = cleaned_value
            return new_dict
        
        elif isinstance(obj, list):
            # Filter out items that contain "other" in specified fields
            new_list = []
            for item in obj:
                # Check if this item should be removed
                if has_other_value(item, field_names):
                    removed_count += 1
                    continue  # Skip this item
                
                # Otherwise, recurse into it
                cleaned_item = recurse(item)
                new_list.append(cleaned_item)
            
            return new_list
        
        else:
            # Primitive value, return as is
            return obj
    
    cleaned_data = recurse(data)
    return cleaned_data, removed_count


def process_json_file(
    input_path: Path,
    output_path: Path,
    field_names: Set[str]
) -> tuple[int, bool]:
    """Process a single JSON file and remove entries with "other" values.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        field_names: Set of field names to check
        
    Returns:
        Tuple of (removed_count, success)
    """
    try:
        # Load JSON
        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        
        # Remove entries with "other"
        cleaned_data, removed_count = remove_other_entries(data, field_names)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(cleaned_data, fh, indent=2, ensure_ascii=False)
        
        return removed_count, True
    
    except Exception as e:
        print(f"\nError processing {input_path}: {e}")
        return 0, False


def process_all_files(input_dir: Path, output_dir: Path, field_names: Set[str]):
    """Process all JSON files in input directory.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory for processed files
        field_names: Set of field names to check
    """
    json_files = sorted(input_dir.rglob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Processing {len(json_files)} JSON files...")
    print(f"Checking fields: {', '.join(sorted(field_names))}")
    print(f"Removing entries where any field value is 'other'\n")
    
    total_removed = 0
    success_count = 0
    
    for json_path in tqdm(json_files, desc="Processing files"):
        # Compute output path (preserve directory structure)
        rel_path = json_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        removed_count, success = process_json_file(json_path, output_path, field_names)
        
        if success:
            success_count += 1
            total_removed += removed_count
    
    print(f"\nDone!")
    print(f"Successfully processed: {success_count}/{len(json_files)} files")
    print(f"Total entries removed: {total_removed}")
    print(f"Output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove entries where specified fields have value 'other'"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing JSON files"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for cleaned JSON files"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    process_all_files(input_dir, output_dir, FIELD_NAMES_TO_CHECK)


if __name__ == "__main__":
    main()


'''
# Basic usage
python scripts/postprocess_remove_other.py \
    --input_dir data/processed_annotations \
    --output_dir data/final_annotations

# Process few-shot examples
python scripts/postprocess_remove_other.py \
    --input_dir data/few-shot-examples \
    --output_dir data/few-shot-examples-cleaned
'''
