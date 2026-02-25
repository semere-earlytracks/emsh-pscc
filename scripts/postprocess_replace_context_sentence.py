#!/usr/bin/env python3
"""Postprocessor to replace contextsentence fields with their exact form from input text.

This script processes JSON files and replaces all contextsentence fields with how they
actually appear in the input text (preserving original case, punctuation, whitespace).

Uses fuzzy matching from align_sentences.py to find the context sentence in the input.
If not found, the contextsentence is set to empty string "".

Usage example:
    python scripts/postprocess_replace_context_sentence.py \
        --input_dir data/processed_annotations \
        --output_dir data/aligned_annotations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional
import sys

from tqdm import tqdm

# Import the fuzzy matching function
sys.path.insert(0, str(Path(__file__).parent))
from align_sentences import find_approx_substring


def replace_context_sentences(
    data: Any,
    input_text: Optional[str]
) -> tuple[Any, int, int, int]:
    """Recursively replace contextsentence fields with their form from input text.
    
    Args:
        data: JSON data (dict, list, or primitive)
        input_text: The input document text to search in
        
    Returns:
        Tuple of (modified data, total_found, not_found_count, skipped_empty_count)
    """
    total_found = 0
    not_found_count = 0
    skipped_empty_count = 0
    
    def recurse(obj: Any) -> Any:
        nonlocal total_found, not_found_count, skipped_empty_count
        
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                if key == "contextsentence":
                    # Handle contextsentence field
                    if isinstance(value, str):
                        if value.strip() == "":
                            # Already empty, skip searching
                            skipped_empty_count += 1
                            new_dict[key] = value
                        elif input_text is None:
                            # No input text available
                            not_found_count += 1
                            new_dict[key] = ""
                        else:
                            # Try to find it in the input text
                            found = find_approx_substring(value, input_text)
                            if found is not None:
                                total_found += 1
                                new_dict[key] = found
                            else:
                                not_found_count += 1
                                new_dict[key] = ""
                    else:
                        # Non-string contextsentence (shouldn't happen, but handle it)
                        new_dict[key] = value
                else:
                    # Recurse into other fields
                    new_dict[key] = recurse(value)
            return new_dict
        
        elif isinstance(obj, list):
            return [recurse(item) for item in obj]
        
        else:
            # Primitive value, return as is
            return obj
    
    modified_data = recurse(data)
    return modified_data, total_found, not_found_count, skipped_empty_count


def process_json_file(
    input_path: Path,
    output_path: Path
) -> tuple[int, int, int, bool]:
    """Process a single JSON file and replace contextsentence fields.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        
    Returns:
        Tuple of (total_found, not_found, skipped_empty, success)
    """
    try:
        # Load JSON
        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        
        # Extract input text if available
        input_text = None
        if isinstance(data, dict) and "input" in data:
            input_text = data.get("input")
            if not isinstance(input_text, str):
                input_text = None
        
        # Replace contextsentence fields
        modified_data, total_found, not_found, skipped = replace_context_sentences(
            data, input_text
        )
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(modified_data, fh, indent=2, ensure_ascii=False)
        
        return total_found, not_found, skipped, True
    
    except Exception as e:
        print(f"\nError processing {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, False


def process_all_files(input_dir: Path, output_dir: Path):
    """Process all JSON files in input directory.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory for processed files
    """
    json_files = sorted(input_dir.rglob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Processing {len(json_files)} JSON files...")
    print(f"Replacing contextsentence fields with their form from input text\n")
    
    total_found_all = 0
    total_not_found_all = 0
    total_skipped_all = 0
    success_count = 0
    
    for json_path in tqdm(json_files, desc="Processing files"):
        # Compute output path (preserve directory structure)
        rel_path = json_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        found, not_found, skipped, success = process_json_file(json_path, output_path)
        
        if success:
            success_count += 1
            total_found_all += found
            total_not_found_all += not_found
            total_skipped_all += skipped
    
    print(f"\nDone!")
    print(f"Successfully processed: {success_count}/{len(json_files)} files")
    print(f"Total contextsentences found and replaced: {total_found_all}")
    print(f"Total contextsentences not found (set to \"\"): {total_not_found_all}")
    print(f"Total empty contextsentences skipped: {total_skipped_all}")
    print(f"Output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace contextsentence fields with their exact form from input text"
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
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    process_all_files(input_dir, output_dir)


if __name__ == "__main__":
    main()


'''
# Basic usage
python scripts/postprocess_replace_context_sentence.py \
    --input_dir data/processed_annotations \
    --output_dir data/aligned_annotations

# Process few-shot examples
python scripts/postprocess_replace_context_sentence.py \
    --input_dir data/few-shot-examples \
    --output_dir data/few-shot-examples-aligned

# Process generated data
python scripts/postprocess_replace_context_sentence.py \
    --input_dir data/generated \
    --output_dir data/generated-aligned
'''
