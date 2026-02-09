"""
Data Annotator Script

This script generates input/output pairs for fine-tuning a model using vLLM.
- Input: text field from clinical documents
- Output: document field (excluding documentid) from structured data

Uses ThreadPool for parallel processing of vLLM API calls via OpenAI Chat Completions API.
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from openai import OpenAI
from pydantic_schema import Document


# Configuration
FEW_SHOT_DIR = Path("data/few-shot-examples")
OUTPUT_DIR = Path("data/generated")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "unsloth/GLM-4.7-Flash-FP8-Dynamic")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))


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


def get_document_schema() -> Dict[str, Any]:
    """
    Get the JSON schema for the Document model (excluding documentid).
    
    Returns:
        JSON schema dictionary
    """
    # Get the full schema
    full_schema = Document.model_json_schema()
    
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
        "content": (
            "You are a medical data extraction assistant. "
            "Given clinical text, extract structured medical information including "
            "patient history, diagnoses, treatments, imaging, biomarkers, and events. \n\n"
            "You MUST return the output as valid JSON following this exact schema (documentid field is optional and can be omitted):\n\n"
            f"{schema_str}\n\n"
            "Important guidelines:\n"
            "- Return ONLY valid JSON, no additional text or explanations\n"
            "- All date fields must be in ISO format (YYYY-MM-DD)\n"
            "- Use empty arrays [] for missing list fields\n"
            "- Include contextsentence field for each extracted entity\n"
            "- Match enum values exactly from the allowed options\n"
            "- Omit the documentid field from your output"
        )
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
    schema_str: str,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generate structured annotation for a given clinical text using vLLM.
    
    Args:
        client: OpenAI client configured for vLLM
        text: Clinical text to annotate
        history_messages: Few-shot examples as message history
        schema_str: JSON schema as a string for response formatting
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
            response_format={"type": "json_object", "schema": schema_str},  # Force JSON output
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
    texts: List[str],
    history_messages: List[Dict[str, str]],
    schema_str: str,
    max_workers: int = MAX_WORKERS
) -> List[Dict[str, Any]]:
    """
    Process multiple texts in parallel using ThreadPoolExecutor.
    
    Args:
        texts: List of clinical texts to annotate
        history_messages: Few-shot examples as message history
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
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_text = {
            executor.submit(
                generate_annotation,
                client,
                text,
                history_messages,
                schema_str,
                {"index": idx}
            ): (idx, text)
            for idx, text in enumerate(texts)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_text):
            idx, text = future_to_text[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed annotation {idx + 1}/{len(texts)}")
            except Exception as e:
                print(f"Error processing text {idx}: {e}")
                results.append({
                    "input": text,
                    "output": None,
                    "error": str(e),
                    "metadata": {"index": idx}
                })
    
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


def main():
    """Main execution function."""
    print("=" * 60)
    print("Data Annotator - Clinical Text Extraction")
    print("=" * 60)
    
    # Get the JSON schema
    schema = get_document_schema()
    schema_str = json.dumps(schema, ensure_ascii=False)

    with open("data/schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False)

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
    
    # Example: Process sample texts (replace with your actual data)
    print("\n3. Processing sample texts...")
    sample_texts = [
        "25/04/2024\n\nPatient presents with chest pain. Diagnosed with myocardial infarction on April 25, 2024. ECG shows ST elevation. Troponin levels elevated. Patient started on aspirin and statin therapy.",
        "10/05/2024\n\nFollow-up consultation. Patient with history of breast cancer diagnosed in 2020. Recent mammogram shows no recurrence. Patient tolerating tamoxifen well. Continue current treatment plan."
    ]
    
    results = process_texts_parallel(sample_texts, history_messages, schema_str)
    
    # Save results
    print("\n4. Saving results...")
    output_file = OUTPUT_DIR / "annotations.json"
    save_results(results, output_file)
    
    print("\n" + "=" * 60)
    print(f"Processing complete! Generated {len(results)} annotations")
    print("=" * 60)


if __name__ == "__main__":
    main()
