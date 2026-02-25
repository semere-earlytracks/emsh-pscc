"""
Data Annotator Script

This script generates input/output pairs for fine-tuning a model using vLLM.
- Input: text field from clinical documents
- Output: document field (excluding documentid) from structured data

Uses ThreadPool for parallel processing of vLLM API calls via OpenAI Chat Completions API.
"""

import json
import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from openai import OpenAI
from pydantic_schema_with_ranges import Document


# Configuration
FEW_SHOT_DIR = Path("data/few-shot-examples")
OUTPUT_DIR = Path("data/generated")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "ig1/medgemma-27b-text-it-FP8-Dynamic")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "64"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "0"))  # 0 means process all files


def load_few_shot_examples() -> List[Dict[str, Any]]:
    """
    Load all few-shot examples from the few-shot-examples directory.
    
    Returns:
        List of dictionaries containing the example data
    """
    examples = []
    
    if not FEW_SHOT_DIR.exists():
        print(f"Warning: Few-shot directory {FEW_SHOT_DIR} does not exist")
        return examples
    
    for json_file in FEW_SHOT_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                examples.append(data)
                print(f"Loaded example: {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return examples


def load_documents_from_directory(directory: Path, num_samples: int = 0) -> List[Dict[str, Any]]:
    """
    Load JSON files recursively from a directory and extract text from 'extract_txt_anon' key.
    Optionally randomly sample n files.
    
    Args:
        directory: Path to the directory containing JSON files
        num_samples: Number of files to randomly sample (0 = all files)
        
    Returns:
        List of dictionaries with 'text', 'source_path', and 'relative_path' keys
    """
    documents = []
    
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist")
        return documents
    
    # Find all JSON files recursively
    json_files = list(directory.rglob("*.json"))
    total_files = len(json_files)
    print(f"Found {total_files} JSON files in {directory}")
    random.seed(42)
    # Randomly sample if num_samples > 0
    if num_samples > 0 and num_samples < total_files:
        json_files = random.sample(json_files, num_samples)
        print(f"Randomly selected {len(json_files)} files")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Extract text from 'extract_txt_anon' key
                if "extract_txt_anon" in data:
                    text = data["extract_txt_anon"]
                    # Append date if available
                    doc_date = data.get("date")
                    if doc_date:
                        text = f"{text}\nDate of the document: {{{doc_date}}}"
                    relative_path = json_file.relative_to(directory)
                    documents.append({
                        "text": text,
                        "source_path": str(json_file),
                        "relative_path": str(relative_path),
                        "original_data": data  # Keep original data for reference
                    })
                else:
                    print(f"Warning: 'extract_txt_anon' key not found in {json_file}")
                    
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(documents)} documents with text")
    return documents


def get_document_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for the Document model (excluding documentid).
    
    Returns:
        JSON schema dictionary
    """
    # Get the full schema
    full_schema = Document.model_json_schema()
    print(f'full_schema: {full_schema}')
    
    # Create a modified schema without documentid as required
    if 'properties' in full_schema and 'documentid' in full_schema['properties']:
        # Make documentid optional by removing it from required fields
        if 'required' in full_schema and 'documentid' in full_schema['required']:
            full_schema['required'] = [r for r in full_schema['required'] if r != 'documentid']
    
    return full_schema


def prepare_history_messages(examples: List[Dict[str, Any]], schema_str: str) -> List[Dict[str, str]]:
    """
    Convert few-shot examples into message history format for the LLM.
    Each example becomes a user message (text) and assistant message (document without documentid).
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        List of message dictionaries in OpenAI chat format
    """
    messages = []
    
    system_message = {
        "role": "system",
        "content": open("data/system_prompt.md", "r", encoding="utf-8").read().format(schema_str=schema_str)
    }
    messages.append(system_message)
    
    for example in examples:
        # User message: the clinical text
        if "text" in example:
            messages.append({
                "role": "user",
                "content": example["text"]
            })
        
        # Assistant message: the structured document without documentid
        if "document" in example:
            document = example["document"].copy()
            # Remove documentid if present
            document.pop("documentid", None)
            
            messages.append({
                "role": "assistant",
                "content": json.dumps(document)
            })
    
    return messages


def generate_annotation(
    client: OpenAI,
    text: str,
    history_messages: List[Dict[str, str]],
    schema: Any,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate structured annotation for a given clinical text using vLLM.
    
    Args:
        client: OpenAI client configured for vLLM
        text: Clinical text to annotate
        history_messages: Few-shot examples as message history
        schema: JSON schema for response formatting
        metadata: Optional metadata to include in the result
        
    Returns:
        Dictionary containing the input text, generated annotation, and metadata
    """
    try:
        # Prepare messages with history + new input
        messages = history_messages.copy()
        messages.append({
            "role": "user",
            "content": text
        })
        
        # Call vLLM via OpenAI-compatible API
        # Try to use JSON mode if available
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            #extra_body={
            #    "guided_json": schema,
            #    # optional: force backend for this request (if your server supports it)
            #    "guided_decoding_backend": "xgrammar",
            #},

            response_format={  # Force JSON output
                "type": "json_schema", 
                "json_schema": {"name": "DocumentSchema", "schema": schema},
            },
        )
        
        # Extract the generated content
        generated_content = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            parsed_document = json.loads(generated_content)
            
            # Validate against Pydantic schema
            try:
                # Add a temporary documentid for validation, then remove it
                validation_doc = parsed_document.copy()
                if 'documentid' not in validation_doc:
                    validation_doc['documentid'] = 'temp-id'
                
                validated = Document.model_validate(validation_doc)
                # Convert back to dict and remove documentid
                parsed_document = validated.to_json_serializable()
                validation_error = None
            except Exception as e:
                print(f"Warning: Validation error: {e}")
                validation_error = str(e)
                # Keep the parsed document even if validation fails
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse generated content as JSON: {e}")
            parsed_document = {"raw_output": generated_content}
            validation_error = f"JSON decode error: {e}"
        
        result = {
            "input": text,
            "output": parsed_document,
            "metadata": metadata or {},
            "model": MODEL_NAME,
            "finish_reason": response.choices[0].finish_reason,
        }
        
        if validation_error:
            result["validation_error"] = validation_error
        
        return result
        
    except Exception as e:
        print(f"Error generating annotation: {e}")
        return {
            "input": text,
            "output": None,
            "error": str(e),
            "metadata": metadata or {}
        }


def process_texts_parallel(
    documents: List[Dict[str, Any]],
    history_messages: List[Dict[str, str]],
    schema: Any,
    output_dir: Path,
    max_workers: int = MAX_WORKERS
) -> List[Dict[str, Any]]:
    """
    Process multiple texts in parallel using ThreadPoolExecutor.
    Saves results as they are completed.
    
    Args:
        documents: List of document dictionaries with 'text' and 'relative_path' keys
        history_messages: Few-shot examples as message history
        schema: JSON schema for response formatting
        output_dir: Output directory to save results
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of annotation results
    """
    # Initialize OpenAI client for vLLM
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
    )
    
    results = []
    saved_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with document metadata
        future_to_doc = {
            executor.submit(
                generate_annotation,
                client,
                doc["text"],
                history_messages,
                schema,
                {"index": idx, "relative_path": doc["relative_path"]}
            ): (idx, doc)
            for idx, doc in enumerate(documents)
        }
        
        # Collect results as they complete and save immediately
        for future in as_completed(future_to_doc):
            idx, doc = future_to_doc[future]
            try:
                result = future.result()
                results.append(result)
                
                # Save result immediately
                relative_path = doc["relative_path"]
                try:
                    output_path = save_result_with_structure(result, output_dir, relative_path)
                    saved_count += 1
                    print(f"✓ [{saved_count}/{len(documents)}] Saved: {relative_path}")
                except Exception as e:
                    print(f"✗ Error saving {relative_path}: {e}")
                
            except Exception as e:
                print(f"✗ Error processing document {idx}: {e}")
                result = {
                    "input": doc.get("text", ""),
                    "output": None,
                    "error": str(e),
                    "metadata": {"index": idx, "relative_path": doc["relative_path"]}
                }
                results.append(result)
                
                # Try to save error result too
                try:
                    save_result_with_structure(result, output_dir, doc["relative_path"])
                    saved_count += 1
                except Exception:
                    pass
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x.get("metadata", {}).get("index", 0))
    
    return results


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """
    Save annotation results to a JSON file.
    
    Args:
        results: List of annotation results
        output_file: Path to output file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def save_result_with_structure(result: Dict[str, Any], output_base_dir: Path, relative_path: str):
    """
    Save a single result maintaining the directory structure.
    
    Args:
        result: Single annotation result
        output_base_dir: Base output directory
        relative_path: Relative path from input directory (e.g., 'subdir/file.json')
    """
    # Parse the relative path
    rel_path_obj = Path(relative_path)
    
    # Use original filename
    output_path = output_base_dir / rel_path_obj
    
    # Create directory structure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return output_path


def main(input_directory: str = None, output_directory: str = None, num_samples: int = None):
    """Main execution function.
    
    Args:
        input_directory: Path to directory containing JSON files to process
        output_directory: Path to output directory for results
        num_samples: Number of files to randomly sample (None/0 = all files)
    """
    print("=" * 60)
    print("Data Annotator - Clinical Text Extraction")
    print("=" * 60)
    
    # Get the JSON schema
    schema = get_document_schema()
    schema_str = json.dumps(schema, ensure_ascii=False)

    schema_file = Path("data/schema.json")
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"Schema saved to {schema_file}")

    # Load few-shot examples
    print("\n1. Loading few-shot examples...")
    examples = load_few_shot_examples()
    print(f"   Loaded {len(examples)} examples")
    
    if not examples:
        print("Error: No few-shot examples found. Exiting.")
        return
    
    # Prepare history messages
    print("\n2. Preparing history messages...")
    history_messages = prepare_history_messages(examples, schema_str)
    print(f"   Prepared {len(history_messages)} messages")
    
    # Load documents from input directory
    print("\n3. Loading documents...")
    input_dir = Path(input_directory) if input_directory else Path("data/input")
    output_dir = Path(output_directory) if output_directory else Path("data/output")
    n_samples = num_samples if num_samples is not None else NUM_SAMPLES
    
    documents = load_documents_from_directory(input_dir, n_samples)
    
    if not documents:
        print("Error: No documents found. Exiting.")
        return
    
    # Process all documents (saving happens inside process_texts_parallel)
    print(f"\n4. Processing {len(documents)} documents...")
    print(f"   Saving results to: {output_dir}")
    print(f"   Results will be saved as they are generated...\n")
    
    results = process_texts_parallel(documents, history_messages, schema, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  - Processed: {len(results)} documents")
    print(f"  - Saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    main(input_dir, output_dir, num_samples)
