#!/usr/bin/env python3
"""End-to-end inference pipeline for clinical text annotation and postprocessing.

This script orchestrates the complete pipeline:
1. data_annotator.py - Generate structured annotations from clinical text
2. infer_pssc_labelV3.py - Map field values to standardized labels
3. infer_biomarkersname.py - Infer biomarker names and filter measurements
4. postprocess_remove_other.py - Remove entries with "other" values
5. postprocess_replace_context_sentence.py - Align context sentences with input text

Usage example:
    python inference_end_to_end.py \
        input_dir=data/input \
        output_dir=data/output \
        num_samples=10
"""
from __future__ import annotations

import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Import main functions from each script
sys.path.insert(0, str(Path(__file__).parent))
from data_annotator import main as data_annotator_main
from merge_predictions_by_patient import process_patient_dir
from copy import deepcopy
from datetime import date

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from scripts.infer_pssc_labelV3 import main as infer_pssc_main
from scripts.infer_biomarkersname import main as infer_biomarkers_main
from scripts.postprocess_remove_other import process_all_files as remove_other_process
from scripts.postprocess_remove_other import FIELD_NAMES_TO_CHECK
from scripts.postprocess_replace_context_sentence import process_all_files as replace_context_process


def convert_ranges_to_onedate(obj):
    """Convert all date ranges to single average dates."""
    def _average_date_str(start_iso: str, end_iso: str) -> str:
        try:
            s = date.fromisoformat(start_iso)
            e = date.fromisoformat(end_iso)
        except Exception:
            return "1900-01-01"
        avg_ord = round((s.toordinal() + e.toordinal()) / 2.0)
        avg = date.fromordinal(int(avg_ord))
        min_date = date(1900, 1, 1)
        if avg < min_date:
            avg = min_date
        return avg.isoformat()
    
    if isinstance(obj, dict):
        # Detect a plain range dict
        if set(obj.keys()) >= {"start", "end"} and len(obj.keys()) == 2:
            return _average_date_str(obj.get("start", ""), obj.get("end", ""))
        new = {}
        for k, v in obj.items():
            new[k] = convert_ranges_to_onedate(v)
        return new
    elif isinstance(obj, list):
        return [convert_ranges_to_onedate(i) for i in obj]
    else:
        return obj


def run_pipeline(
    cfg: DictConfig
):
    """Run the complete end-to-end inference pipeline.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert paths to Path objects
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    labels_dir = Path(cfg.pssc_labels.labels_dir)
    cache_dir = Path(cfg.pssc_labels.cache_dir)
    
    print("=" * 80)
    print("END-TO-END INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 80)
    
    # Set environment variables for data_annotator
    if cfg.data_annotator.vllm_base_url:
        os.environ["VLLM_BASE_URL"] = cfg.data_annotator.vllm_base_url
    if cfg.data_annotator.vllm_api_key:
        os.environ["VLLM_API_KEY"] = cfg.data_annotator.vllm_api_key
    if cfg.data_annotator.model_name:
        os.environ["VLLM_MODEL_NAME"] = cfg.data_annotator.model_name
    os.environ["MAX_WORKERS"] = str(cfg.data_annotator.max_workers)
    os.environ["TEMPERATURE"] = str(cfg.data_annotator.temperature)
    os.environ["MAX_TOKENS"] = str(cfg.data_annotator.max_tokens)
    os.environ["DISABLE_FEW_SHOT"] = str(cfg.data_annotator.disable_few_shot).lower()
    os.environ["USE_RESPONSE_FORMAT"] = str(cfg.data_annotator.use_response_format).lower()
    
    # Create temporary directories for intermediate results
    temp_base = tempfile.mkdtemp(prefix="inference_pipeline_")
    print(f"\nTemporary directory: {temp_base}")
    
    try:
        temp_dir_1 = Path(temp_base) / "step1_annotated"
        temp_dir_2 = Path(temp_base) / "step2_pssc_labels"
        temp_dir_3 = Path(temp_base) / "step3_biomarkers"
        temp_dir_4 = Path(temp_base) / "step4_no_other"
        temp_dir_5 = Path(temp_base) / "step5_context_aligned"
        
        # Step 1: Generate annotations
        print("\n" + "=" * 80)
        print("STEP 1: Generating structured annotations (data_annotator)")
        print("=" * 80)
        
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        # Set up args for data_annotator
        sys.argv = [
            "data_annotator.py",
            str(input_dir),
            str(temp_dir_1),
        ]
        if cfg.num_samples:
            sys.argv.append(str(cfg.num_samples))
        
        data_annotator_main(str(input_dir), str(temp_dir_1), cfg.num_samples)
        
        # Restore original argv
        sys.argv = original_argv
        
        print(f"✓ Step 1 complete: {temp_dir_1}")
        
        # Step 2: Infer PSSC labels
        print("\n" + "=" * 80)
        print("STEP 2: Inferring PSSC labels (infer_pssc_labelV3)")
        print("=" * 80)
        
        # Set up args for infer_pssc_labelV3
        sys.argv = [
            "infer_pssc_labelV3.py",
            "--model", cfg.pssc_labels.model,
            "--input_dir", str(temp_dir_1),
            "--output_dir", str(temp_dir_2),
            "--labels_dir", str(labels_dir),
            "--cache_dir", str(cache_dir),
            "--batch_size", str(cfg.pssc_labels.batch_size),
        ]
        if cfg.pssc_labels.device:
            sys.argv.extend(["--device", cfg.pssc_labels.device])
        if cfg.pssc_labels.skip_pass1:
            sys.argv.append("--skip_pass1")
        if cfg.pssc_labels.strip_codes:
            sys.argv.append("--strip_codes")
        if cfg.pssc_labels.map_threshold:
            sys.argv.extend(["--map_threshold", str(cfg.pssc_labels.map_threshold)])
        
        infer_pssc_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print(f"✓ Step 2 complete: {temp_dir_2}")
        
        # Step 3: Infer biomarkers and filter measurements
        print("\n" + "=" * 80)
        print("STEP 3: Inferring biomarkers and filtering measurements (infer_biomarkersname)")
        print("=" * 80)
        
        # Set up args for infer_biomarkersname
        biomarker_csv = Path(cfg.biomarkers.labels_csv)
        sys.argv = [
            "infer_biomarkersname.py",
            "--input_dir", str(temp_dir_2),
            "--output_dir", str(temp_dir_3),
            "--labels_csv", str(biomarker_csv),
            "--cache_dir", str(cfg.biomarkers.cache_dir),
        ]
        if cfg.biomarkers.case_sensitive:
            sys.argv.append("--case_sensitive")
        if cfg.biomarkers.skip_pass1:
            sys.argv.append("--skip_pass1")
        if cfg.biomarkers.no_biomarker_inference:
            sys.argv.append("--no-biomarker-inference")
        
        infer_biomarkers_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print(f"✓ Step 3 complete: {temp_dir_3}")
        
        # Step 4: Remove entries with "other" values
        if cfg.remove_other.enabled:
            print("\n" + "=" * 80)
            print("STEP 4: Removing entries with 'other' values (postprocess_remove_other)")
            print("=" * 80)
            
            remove_other_process(temp_dir_3, temp_dir_4, FIELD_NAMES_TO_CHECK)
            
            print(f"✓ Step 4 complete: {temp_dir_4}")
            prev_output = temp_dir_4
        else:
            print("\n⊘ Skipping Step 4: Remove 'other' entries (disabled in config)")
            prev_output = temp_dir_3
        
        # Step 5: Replace context sentences
        if cfg.replace_context.enabled:
            print("\n" + "=" * 80)
            print("STEP 5: Aligning context sentences (postprocess_replace_context_sentence)")
            print("=" * 80)
            
            replace_context_process(prev_output, temp_dir_5)
            
            print(f"✓ Step 5 complete: {temp_dir_5}")
            prev_output = temp_dir_5
        else:
            print("\n⊘ Skipping Step 5: Replace context sentences (disabled in config)")
        
        # Step 6: Merge predictions by patient
        if cfg.merge_by_patient.enabled:
            print("\n" + "=" * 80)
            print("STEP 6: Merging predictions by patient (merge_predictions_by_patient)")
            print("=" * 80)
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            documents = []
            patients = []
            
            # Process each patient directory
            for child in sorted(prev_output.iterdir()):
                if child.is_dir():
                    process_patient_dir(child, documents, patients)
            
            # Create merged output with correct keys (singular: patient, document)
            merged_output = {"patient": patients, "document": documents}
            
            # Create onedate version (this will be the final prediction.json)
            onedate_output = convert_ranges_to_onedate(deepcopy(merged_output))
            
            # Write the onedate version as prediction.json (final output)
            final_output_path = output_dir / cfg.final_merged_json
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump(onedate_output, f, indent=2, ensure_ascii=False)
            
            print(f"  Final output (onedate) written to: {final_output_path}")
            print(f"  Total patients: {len(patients)}")
            print(f"  Total documents: {len(documents)}")
            
            # Optionally write the ranges version with a different name
            ranges_filename = final_output_path.stem + ".ranges.json"
            ranges_path = output_dir / ranges_filename
            with open(ranges_path, "w", encoding="utf-8") as f:
                json.dump(merged_output, f, indent=2, ensure_ascii=False)
            
            print(f"  Ranges version written to: {ranges_path}")
            
            print(f"✓ Step 6 complete: {final_output_path} (onedate) and {ranges_path} (ranges)")
        else:
            print("\n⊘ Skipping Step 6: Merge by patient (disabled in config)")
            # Copy final output to output_dir
            print(f"\nCopying final results to: {output_dir}")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(prev_output, output_dir)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        if cfg.merge_by_patient.enabled:
            print(f"Final merged output (onedate): {output_dir / cfg.final_merged_json}")
            print(f"Ranges version: {output_dir / (Path(cfg.final_merged_json).stem + '.ranges.json')}")
        else:
            print(f"Final output directory: {output_dir}")
        
    finally:
        # Clean up temporary directories unless keep_temp is True
        if cfg.keep_temp:
            print(f"\n⚠ Temporary files preserved at: {temp_base}")
        else:
            print(f"\nCleaning up temporary directory: {temp_base}")
            shutil.rmtree(temp_base, ignore_errors=True)


@hydra.main(version_base=None, config_path="conf", config_name="config_bigmodel")
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object loaded from config_bigmodel.yaml
    """
    # Validate inputs
    input_dir = Path(cfg.input_dir)
    labels_dir = Path(cfg.pssc_labels.labels_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Run the pipeline
    run_pipeline(cfg)


if __name__ == "__main__":
    main()


'''
# Basic usage (uses defaults from config_bigmodel.yaml)
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output

# Use fast model configuration (fine-tuned, no few-shot)
python inference_end_to_end.py \
    --config-name=config_fastmodel \
    input_dir=data/input \
    output_dir=data/output

# Process subset of files with big model
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    num_samples=10

# Use GPU and larger batch size (with big model)
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    pssc_labels.device=cuda \
    pssc_labels.batch_size=256

# Use fast model with custom settings
python inference_end_to_end.py \
    --config-name=config_fastmodel \
    input_dir=data/input \
    output_dir=data/output \
    pscc_labels.device=cuda \
    num_samples=100

# Keep temporary files for debugging
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    keep_temp=true

# Disable certain steps
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    remove_other.enabled=false \
    merge_by_patient.enabled=false

# Full example with all options (big model)
python inference_end_to_end.py \
    input_dir=data/raw_documents \
    output_dir=data/final_annotations \
    num_samples=100 \
    final_merged_json=my_predictions.json \
    pssc_labels.model=all-MiniLM-L6-v2 \
    pssc_labels.device=cuda \
    pssc_labels.batch_size=256 \
    pssc_labels.strip_codes=false \
    biomarkers.case_sensitive=true \
    keep_temp=true

# Override vLLM settings
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    data_annotator.vllm_base_url=http://localhost:8001/v1 \
    data_annotator.max_workers=32

# Disable few-shot examples (with big model)
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    data_annotator.disable_few_shot=true
'''
