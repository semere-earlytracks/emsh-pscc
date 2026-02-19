#!/usr/bin/env python3
"""Create conversation dataset for LLM training from input/output JSON files.

Reads all .json files in the input directory. Each file must contain 'input' and 'output' keys.
Creates a dataset_train.json file containing an array of objects with a 'conversation' key.
Each conversation is an array of messages in OpenAI chat format:
- User message: the 'input' text
- Assistant message: the 'output' (as JSON string)

Usage:
    python scripts/create_conversation_dataset.py \
        --input_dir data/generated \
        --output_file data/dataset_train.json
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from pydantic_schema_with_ranges import Document


def get_document_schema() -> Dict[str, Any]:
    """Get the JSON schema for the Document model (excluding documentid)."""
    full_schema = Document.model_json_schema()
    
    # Make documentid optional by removing it from required fields
    if 'properties' in full_schema and 'documentid' in full_schema['properties']:
        if 'required' in full_schema and 'documentid' in full_schema['required']:
            full_schema['required'] = [r for r in full_schema['required'] if r != 'documentid']
    
    return full_schema


def load_system_prompt(schema_str: str) -> str:
    """Load and format system prompt with schema."""
    system_prompt_path = Path("data/system_prompt.md")
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
    
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    return system_prompt.format(schema_str=schema_str)


def load_input_output_jsons(input_dir: Path) -> List[Dict[str, Any]]:
    """Load all .json files with 'input' and 'output' keys from input_dir."""
    examples = []
    json_files = sorted(input_dir.rglob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Accept both single dict or list of dicts
                if isinstance(data, dict) and 'input' in data and 'output' in data:
                    examples.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'input' in item and 'output' in item:
                            examples.append(item)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    print(f"Loaded {len(examples)} input/output pairs from {len(json_files)} files.")
    return examples


def create_conversation_dataset(examples: List[Dict[str, Any]], system_message: str) -> List[Dict[str, Any]]:
    """Create conversation dataset in OpenAI chat format with system message."""
    dataset = []
    for ex in examples:
        input_text = ex['input']
        output_data = ex['output']
        # Convert output to JSON string if not already
        if isinstance(output_data, dict):
            output_str = json.dumps(output_data, ensure_ascii=False)
        else:
            output_str = str(output_data)
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_str}
        ]
        dataset.append({"conversation": conversation})
    return dataset


def save_dataset(dataset: List[Dict[str, Any]], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create conversation dataset for LLM training.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing .json files with input/output keys"
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output file for conversation dataset (e.g., data/dataset_train.json)"
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get schema and prepare system message
    print("Loading schema and system prompt...")
    schema = get_document_schema()
    schema_str = json.dumps(schema, ensure_ascii=False)
    system_message = load_system_prompt(schema_str)
    print(f"System message length: {len(system_message)} characters")
    
    examples = load_input_output_jsons(input_dir)
    dataset = create_conversation_dataset(examples, system_message)


if __name__ == "__main__":
    main()
