#!/usr/bin/env python3
"""Process JSON files and replace biomarkername field values using exact string matching.

This script operates in two passes:
1. Pass 1: Collect all unique strings from biomarkername fields across all JSON files,
   map them using exact string matching against biomarkername_ext.csv, and save mappings to disk.
2. Pass 2: Read JSON files again and replace field values using the saved mappings.
   Additionally, filters measuretype entries (Height in cms, Weight) by validating
   measurevalue against regex-extracted values from contextsentence.

Can also run with --no-biomarker-inference to skip biomarkername inference and only apply measure filtering.

Unlike the embedding-based approach, this uses exact string matching for biomarker names.

Usage example:
    python scripts/infer_biomarkersname.py \
        --input_dir data/raw_annotations \
        --output_dir data/processed_annotations \
        --labels_csv data/pssc-labels/biomarkername_ext.csv \
        --cache_dir data/label_cache
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, Set, Optional
from collections import defaultdict

from tqdm import tqdm


# Regex patterns for weight and height extraction
WEIGHT_REGEX = re.compile(
    r"""
    (?ix)
    \b
    (?:
        poids|pds|p\.?\s*d\.?\s*s\.?|pese|pèse|pesant|pesait
    )?
    \s*[:=]?\s*
    (?P<value>
        \d{1,3}
        (?:[.,]\d{1,2})?
        (?:\s*-\s*\d{1,3}(?:[.,]\d{1,2})?)?
    )
    \s*
    (?P<unit>
        kgs?|kilos?|kilogrammes?
    )
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

HEIGHT_REGEX = re.compile(
    r"""
    (?ix)                                   # i: ignorecase, x: verbose
    \b
    (?:
        # =========================
        # A) Contexted (safer): Taille / T: / Height ...
        #    -> allows unitless cm like "Taille 160" or "T: 167"
        # =========================
        (?:
            (?P<label>
                taille|taill?e|height|t
            )
            \s*
            (?:\(?\s*en\s*cm\s*\)?)?        # optional "(en cm)"
            \s*[:=]?\s*
        )
        (?:
            # A1) unitless cm (ONLY because we have the label)
            (?P<cm_ctx>
                (?:1[2-9]\d|2[0-2]\d)       # 120..229
                (?:[.,]\d+)?                # allow 169.0
            )
            (?!\s*(?:kg|bpm|/|\d))          # reduce obvious non-heights
            \b
            |
            # A2) explicit cm with unit
            (?P<cm_ctx_unit>
                (?:1[2-9]\d|2[0-2]\d)
                (?:[.,]\d+)?
            )
            \s*(?:cm|cms|centim(?:e|è)tres?)\b
            |
            # A3) meters like 1,70 m or 1.74m or 1 74 m
            (?:
                (?P<m_ctx_int>[1lI])\s*
                [.,]?\s*
                (?P<m_ctx_dec>\d{1,2})
                \s*(?:m|m[eè]tre|metre)s?
                (?!\s*[2²])                 # avoid m2
            )
            |
            # A4) meters like 1m61 / 1 m 61 / 1M6O / lm90
            (?:
                (?P<m_ctx2_int>[1lI])\s*
                (?:m|M|m[eè]tre|metre)\s*
                (?P<m_ctx2_cm>[0-9O]{2})
                (?!\s*[2²])                 # avoid m2
            )
        )

        |

        # =========================
        # B) No context label: only accept if unit is present
        # =========================
        (?:
            # B1) explicit centimeters
            (?P<cm>
                (?:1[2-9]\d|2[0-2]\d)
                (?:[.,]\d+)?
            )
            \s*(?:cm|cms|centim(?:e|è)tres?)\b
            |
            # B2) meters like 1,70 m / 1.74m / 1 74 m
            (?:
                (?P<m_int>[1lI])\s*
                [.,]?\s*
                (?P<m_dec>\d{1,2})
                \s*(?:m|m[eè]tre|metre)s?
                (?!\s*[2²])                 # avoid m2
            )
            |
            # B3) meters like 1m61 / 1 m 61 / 1M6O / lm90
            (?:
                (?P<m2_int>[1lI])\s*
                (?:m|M|m[eè]tre|metre)\s*
                (?P<m2_cm>[0-9O]{2})
                (?!\s*[2²])                 # avoid m2
            )
        )
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def extract_weights(text: str):
    """
    Returns a list of weight values (floats) extracted from the text.
    Keeps only plausible human weights (10..250 kg) to drop noise.
    """
    out = []
    for m in WEIGHT_REGEX.finditer(text):
        value_raw = m.group("value")
        # Skip range values (e.g., "50-60")
        if "-" in value_raw:
            continue
        # Normalize value: French comma -> dot
        v = float(value_raw.replace(",", "."))
        # Only keep plausible human weights
        if 10 <= v <= 250:
            out.append(v)
    return out


def extract_heights_cm(text: str):
    """
    Returns heights in centimeters as ints.
    Keeps only plausible human heights (120..230 cm) to drop noise like temps, Hb, etc.
    """
    out = []
    for m in HEIGHT_REGEX.finditer(text):
        cm_val = None
        if m.group("cm_ctx") or m.group("cm_ctx_unit"):
            raw_num = (m.group("cm_ctx") or m.group("cm_ctx_unit")).replace(",", ".")
            cm_val = int(round(float(raw_num)))
        elif m.group("m_ctx_int") and m.group("m_ctx_dec"):
            i = m.group("m_ctx_int").replace("l", "1").replace("I", "1")
            cm_val = int(i) * 100 + int(m.group("m_ctx_dec"))
        elif m.group("m_ctx2_int") and m.group("m_ctx2_cm"):
            i = m.group("m_ctx2_int").replace("l", "1").replace("I", "1")
            cm_part = m.group("m_ctx2_cm").replace("O", "0")
            cm_val = int(i) * 100 + int(cm_part)
        elif m.group("cm"):
            cm_val = int(round(float(m.group("cm").replace(",", "."))))
        elif m.group("m_int") and m.group("m_dec"):
            i = m.group("m_int").replace("l", "1").replace("I", "1")
            cm_val = int(i) * 100 + int(m.group("m_dec"))
        elif m.group("m2_int") and m.group("m2_cm"):
            i = m.group("m2_int").replace("l", "1").replace("I", "1")
            cm_part = m.group("m2_cm").replace("O", "0")
            cm_val = int(i) * 100 + int(cm_part)
        if cm_val is not None and 120 <= cm_val <= 230:
            out.append(cm_val)
    return out


def find_word_boundary_match(
    string: str,
    biomarker_lookup: Dict[str, str],
    case_sensitive: bool = False
) -> Optional[str]:
    """Find a biomarker match using word boundaries.
    
    Tries exact match first, then word-boundary partial match.
    For example, "PR" matches "PR positive" but not "PRESCRIBE".
    
    Args:
        string: String to match
        biomarker_lookup: Text-to-label lookup from CSV
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        The label if matched, None otherwise
    """
    # Try exact match first (highest priority)
    if case_sensitive:
        if string in biomarker_lookup:
            return biomarker_lookup[string]
    else:
        string_lower = string.lower()
        case_insensitive_lookup = {k.lower(): v for k, v in biomarker_lookup.items()}
        if string_lower in case_insensitive_lookup:
            return case_insensitive_lookup[string_lower]
    
    # Try word-boundary matching with each lookup term
    # Sort by length descending to prefer longer matches
    sorted_terms = sorted(biomarker_lookup.items(), key=lambda x: len(x[0]), reverse=True)
    
    for term, label in sorted_terms:
        # Create regex pattern with word boundaries
        if case_sensitive:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, string):
                return label
        else:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, string, re.IGNORECASE):
                return label
    
    return None


def load_biomarker_lookup(csv_path: Path) -> Dict[str, str]:
    """Load biomarker text-to-label mapping from CSV file.
    
    Args:
        csv_path: Path to biomarkername_ext.csv
        
    Returns:
        Dictionary mapping text values to labels
    """
    lookup = {}
    
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            if text and label:
                # Store both as-is and case-insensitive versions
                lookup[text] = label
    
    return lookup


def collect_biomarkername_values(data: Any) -> Set[str]:
    """Recursively collect unique biomarkername values from JSON data.
    
    Args:
        data: JSON data (dict, list, or primitive)
        
    Returns:
        Set of unique non-empty biomarkername values
    """
    collected = set()
    
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "biomarkername":
                    # Collect the value if it's a non-empty string
                    if value and isinstance(value, str) and value.strip():
                        collected.add(value.strip())
                recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(data)
    return collected


def collect_all_biomarker_strings(input_dir: Path) -> Set[str]:
    """Pass 1: Collect all unique biomarkername strings from all JSON files.
    
    Args:
        input_dir: Directory containing JSON files
        
    Returns:
        Set of unique biomarkername strings
    """
    json_files = sorted(input_dir.rglob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return set()
    
    print(f"Pass 1: Collecting biomarkername strings from {len(json_files)} JSON files...")
    
    all_collected = set()
    
    for json_path in tqdm(json_files, desc="Collecting strings"):
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            collected = collect_biomarkername_values(data)
            all_collected.update(collected)
        
        except Exception as e:
            print(f"\\nError reading {json_path}: {e}")
            continue
    
    return all_collected


def create_biomarker_mapping(
    collected_strings: Set[str],
    biomarker_lookup: Dict[str, str],
    cache_dir: Path,
    case_sensitive: bool = False
) -> Dict[str, str]:
    """Create and save biomarkername string-to-label mapping using word-boundary matching.
    
    Args:
        collected_strings: Set of unique biomarkername strings from JSON files
        biomarker_lookup: Text-to-label lookup from CSV
        cache_dir: Directory to save mapping file
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        Dictionary mapping strings to labels
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\\nCreating biomarkername mappings...")
    print(f"Total unique biomarkername values: {len(collected_strings)}")
    print(f"Case-sensitive matching: {case_sensitive}")
    print(f"Using word-boundary matching (e.g., 'PR' matches 'PR positive' but not 'PRESCRIBE')")
    
    # Create mapping
    mapping = {}
    matched_count = 0
    unmatched_values = []
    
    for string in sorted(collected_strings):
        label = find_word_boundary_match(string, biomarker_lookup, case_sensitive)
        
        if label:
            mapping[string] = label
            matched_count += 1
        else:
            mapping[string] = "other"
            unmatched_values.append(string)
    
    # Save mapping
    cache_path = cache_dir / "biomarkername_mapping.json"
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)
    
    print(f"  Saved mapping to: {cache_path}")
    print(f"  Matched: {matched_count}/{len(collected_strings)}")
    print(f"  Unmatched (mapped to 'other'): {len(unmatched_values)}")
    
    if unmatched_values and len(unmatched_values) <= 20:
        print(f"  Unmatched values: {', '.join(unmatched_values)}")
    elif unmatched_values:
        print(f"  First 20 unmatched values: {', '.join(unmatched_values[:20])}")
    
    return mapping


def load_biomarker_mapping(cache_dir: Path) -> Dict[str, str]:
    """Load biomarkername mapping from cache directory.
    
    Args:
        cache_dir: Directory containing mapping file
        
    Returns:
        Dictionary mapping strings to labels
    """
    cache_path = cache_dir / "biomarkername_mapping.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {cache_path}")
    
    with cache_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_and_process_new_biomarkers(
    input_dir: Path,
    existing_mapping: Dict[str, str],
    biomarker_lookup: Dict[str, str],
    cache_dir: Path,
    case_sensitive: bool = False
) -> Dict[str, str]:
    """Find biomarkername strings not in existing mapping and process them.
    
    Args:
        input_dir: Directory containing JSON files
        existing_mapping: Existing string-to-label mapping
        biomarker_lookup: Text-to-label lookup from CSV
        cache_dir: Directory to update mapping file
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        Updated mapping with new strings added
    """
    print("\\nChecking for new biomarkername strings not in cached mapping...")
    collected_strings = collect_all_biomarker_strings(input_dir)
    
    # Find new strings
    new_strings = collected_strings - set(existing_mapping.keys())
    
    if not new_strings:
        print("No new biomarkername strings found. All values are already in cache.")
        return existing_mapping
    
    print(f"\\nFound {len(new_strings)} new biomarkername strings to process")
    print(f"Case-sensitive matching: {case_sensitive}")
    print(f"Using word-boundary matching (e.g., 'PR' matches 'PR positive' but not 'PRESCRIBE')")
    
    # Process new strings
    new_mapping = {}
    matched_count = 0
    unmatched_values = []
    
    for string in new_strings:
        label = find_word_boundary_match(string, biomarker_lookup, case_sensitive)
        
        if label:
            new_mapping[string] = label
            matched_count += 1
        else:
            new_mapping[string] = "other"
            unmatched_values.append(string)
    
    # Merge with existing mapping
    updated_mapping = existing_mapping.copy()
    updated_mapping.update(new_mapping)
    
    # Save updated mapping
    cache_path = cache_dir / "biomarkername_mapping.json"
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(updated_mapping, fh, indent=2, ensure_ascii=False)
    
    print(f"  Updated mapping saved to: {cache_path}")
    print(f"  Total mappings: {len(updated_mapping)} (added {len(new_mapping)})")
    print(f"  New strings matched: {matched_count}/{len(new_strings)}")
    print(f"  New strings unmatched: {len(unmatched_values)}")
    
    if unmatched_values and len(unmatched_values) <= 10:
        print(f"  Unmatched new values: {', '.join(unmatched_values)}")
    
    return updated_mapping


def replace_biomarkername_values(data: Any, mapping: Dict[str, str]):
    """Recursively replace biomarkername values in JSON data using mapping.
    
    Args:
        data: JSON data (dict, list, or primitive) - modified in place
        mapping: Dictionary mapping strings to labels
    """
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if key == "biomarkername" and isinstance(value, str):
                    # Replace with mapped label if available
                    stripped_value = value.strip()
                    if stripped_value in mapping:
                        obj[key] = mapping[stripped_value]
                else:
                    recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(data)


def filter_measure_entries(data: Any):
    """Recursively filter measuretype entries based on regex validation.
    
    For entries with measuretype "Height in cms" or "Weight", extracts values
    from contextsentence using regex and validates against measurevalue.
    Removes entries where measurevalue doesn't match extracted values (within 0.1 margin).
    
    Args:
        data: JSON data (dict, list, or primitive) - modified in place
    """
    def should_keep_entry(entry: dict) -> bool:
        """Check if a measurement entry should be kept based on regex validation."""
        if not isinstance(entry, dict):
            return True
        
        measuretype = entry.get("measuretype", "")
        measurevalue = entry.get("measurevalue", "")
        contextsentence = entry.get("contextsentence", "")
        
        # Only validate Height in cms and Weight
        if measuretype not in ["Height in cms", "Weight"]:
            return True
        
        if not contextsentence or not measurevalue:
            return True  # Keep entries without context or value
        
        try:
            if measuretype == "Height in cms":
                # Extract heights from context
                extracted_heights = extract_heights_cm(contextsentence)
                if not extracted_heights:
                    return False  # No heights found in context
                
                # Check if measurevalue matches any extracted height (with margin)
                measure_val = float(measurevalue)
                for height in extracted_heights:
                    if abs(height - measure_val) <= 0.1:
                        return True
                return False
            
            elif measuretype == "Weight":
                # Extract weights from context
                extracted_weights = extract_weights(contextsentence)
                if not extracted_weights:
                    return False  # No weights found in context
                
                # Check if measurevalue matches any extracted weight (with margin)
                measure_val = float(measurevalue)
                for weight in extracted_weights:
                    if isinstance(weight, (int, float)):
                        if abs(weight - measure_val) <= 0.1:
                            return True
                return False
        
        except (ValueError, TypeError):
            # If we can't parse measurevalue, keep the entry
            return True
        
        return True
    
    def recurse(obj):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(value, list):
                    # Filter list items that look like measurement entries
                    filtered_list = []
                    for item in value:
                        if isinstance(item, dict) and "measuretype" in item:
                            if should_keep_entry(item):
                                recurse(item)
                                filtered_list.append(item)
                        else:
                            recurse(item)
                            filtered_list.append(item)
                    obj[key] = filtered_list
                else:
                    recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
    
    recurse(data)


def process_all_files(
    input_dir: Path,
    output_dir: Path,
    mapping: Optional[Dict[str, str]] = None
):
    """Pass 2: Process all JSON files and replace biomarkername values using mapping.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory for processed files
        mapping: Dictionary mapping strings to labels (optional)
    """
    json_files = sorted(input_dir.rglob("*.json"))
    
    desc = "Processing files" if mapping is None else "Replacing biomarkername values"
    print(f"\\nPass 2: Processing {len(json_files)} JSON files...")
    
    for json_path in tqdm(json_files, desc=desc):
        try:
            # Load JSON
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            # Replace values (only if mapping is provided)
            if mapping is not None:
                replace_biomarkername_values(data, mapping)
            
            # Filter measuretype entries based on regex validation
            filter_measure_entries(data)
            
            # Compute output path (preserve directory structure)
            rel_path = json_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Write output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"\\nError processing {json_path}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Process JSON files using exact string matching for biomarkername fields"
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
        "--labels_csv",
        default="data/pssc-labels/biomarkername_ext.csv",
        help="Path to biomarkername_ext.csv with label and text columns (default: data/pssc-labels/biomarkername_ext.csv)"
    )
    parser.add_argument(
        "--cache_dir",
        default="data/label_cache",
        help="Directory to store string-to-label mappings (default: data/label_cache)"
    )
    parser.add_argument(
        "--skip_pass1",
        action="store_true",
        help="Skip Pass 1 (use existing mappings in cache_dir)"
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="Use case-sensitive matching (default: case-insensitive)"
    )
    parser.add_argument(
        "--no-biomarker-inference",
        action="store_true",
        help="Skip biomarkername inference and only apply measure filtering"
    )
    args = parser.parse_args()
    
    # Setup
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    labels_csv = Path(args.labels_csv)
    cache_dir = Path(args.cache_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Skip biomarker inference if flag is set
    if args.no_biomarker_inference:
        print("Skipping biomarkername inference (--no-biomarker-inference flag set)")
        print("Only applying measure filtering...")
        process_all_files(input_dir, output_dir, mapping=None)
        print(f"\nDone! Processed files written to {output_dir}")
        return
    
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    
    print(f"Using labels CSV: {labels_csv}")
    print(f"Case-sensitive matching: {args.case_sensitive}")
    
    # Load biomarker lookup from CSV
    biomarker_lookup = load_biomarker_lookup(labels_csv)
    print(f"Loaded {len(biomarker_lookup)} biomarker entries from CSV")
    
    # Pass 1: Collect and map strings
    if not args.skip_pass1:
        collected_strings = collect_all_biomarker_strings(input_dir)
        
        print(f"\\nCollected {len(collected_strings)} unique biomarkername values")
        
        # Create and save mapping
        mapping = create_biomarker_mapping(
            collected_strings,
            biomarker_lookup,
            cache_dir,
            case_sensitive=args.case_sensitive
        )
    else:
        print("Skipping Pass 1 (using existing mappings)")
        mapping = load_biomarker_mapping(cache_dir)
    
    # Load existing mapping and check for new strings
    if not args.skip_pass1:
        # Already have the mapping from Pass 1
        pass
    else:
        mapping = load_biomarker_mapping(cache_dir)
    
    print(f"\\nLoaded {len(mapping)} biomarkername mappings from cache")
    
    # Check for new strings not in mappings and process them
    updated_mapping = find_and_process_new_biomarkers(
        input_dir,
        mapping,
        biomarker_lookup,
        cache_dir,
        case_sensitive=args.case_sensitive
    )
    
    # Process all files
    process_all_files(
        input_dir,
        output_dir,
        updated_mapping
    )
    
    print(f"\\nDone! Processed files written to {output_dir}")


if __name__ == "__main__":
    main()


'''
# Full run (both passes)
python scripts/infer_biomarkersname.py \
    --input_dir data/raw_annotations \
    --output_dir data/processed_annotations \
    --labels_csv data/pssc-labels/biomarkername_ext.csv \
    --cache_dir data/label_cache

# Case-sensitive matching
python scripts/infer_biomarkersname.py \
    --input_dir data/raw_annotations \
    --output_dir data/processed_annotations \
    --case_sensitive

# Skip Pass 1 (use existing mappings)
python scripts/infer_biomarkersname.py \
    --input_dir data/raw_annotations \
    --output_dir data/processed_annotations \
    --cache_dir data/label_cache \
    --skip_pass1

# Skip biomarker inference (only apply measure filtering)
python scripts/infer_biomarkersname.py \
    --input_dir data/raw_annotations \
    --output_dir data/processed_annotations \
    --no-biomarker-inference
'''
