# EMSH LLM Data Generation Pipeline

An end-to-end pipeline for annotating clinical text from patient documents using Large Language Models (LLMs). The pipeline generates structured JSON annotations from unstructured clinical text, maps values to standardized labels, and performs post-processing and merging operations.

This project uses a **uv-managed** Python environment and is distributed as a **zip archive**.

---

## Table of Contents

- [Setup](#setup)
- [Input Data Structure](#input-data-structure)
- [Running the Pipeline](#running-the-pipeline)
  - [Mode 1: Foundation Model (medgemma-27b)](#mode-1-foundation-model-medgemma-27b)
  - [Mode 2: Fine-tuned Model (medgemma-4b)](#mode-2-fine-tuned-model-medgemma-4b)
- [Output](#output)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Unzip the codebase

```bash
unzip codebase.zip
cd emsh_llm_datagen
```

### 2. Activate the virtual environment

The codebase includes a pre-configured uv virtual environment:

```bash
source .venv/bin/activate
```

---

## Input Data Structure

The input directory should contain **patient folders**, where each patient folder contains one or more **JSON document files**:

```
input/
├── patient_001/
│   ├── document_A.json
│   ├── document_B.json
│   └── document_C.json
├── patient_002/
│   ├── document_X.json
│   └── document_Y.json
└── ...
```

Each JSON file should contain clinical text to be annotated.

---

## Running the Pipeline

The pipeline can be run in two modes, depending on your quality/speed requirements:

### Mode 1: Foundation Model (medgemma-27b)

**Best for:** Higher quality annotations (but slower)

#### Step 1: Start the vLLM server

```bash
vllm serve ig1/medgemma-27b-text-it-FP8-Dynamic
```

#### Step 2: Run the annotation pipeline

```bash
python inference_end_to_end.py \
    --config-name=config_bigmodel \
    input_dir=data/input \
    output_dir=data/output
```

---

### Mode 2: Fine-tuned Model (medgemma-4b)

**Best for:** Faster processing with acceptable quality

#### Step 1: Start the vLLM server

```bash
vllm serve model/fine-tuned-model \
    --served-model-name merged16 \
    --quantization="fp8"
```

#### Step 2: Run the annotation pipeline

```bash
python inference_end_to_end.py \
    --config-name=config_fastmodel \
    input_dir=data/input \
    output_dir=data/output
```

---

## Output

The pipeline generates a single merged output file:

```
output/
└── prediction.json
```

### Output Structure

The `prediction.json` file has the following structure:

```json
{
  "patient": [
    {
      "patientid": "patient_001",
      ...
    }
  ],
  "document": [
    {
      "documentid": "document_A",
      ...
    }
  ]
}
```

- **`patient`**: Array of patient-level merged data (concatenates and deduplicates information from all documents for each patient)
- **`document`**: Array of document-level annotations (one entry per input JSON file)
- All date ranges are converted to single average dates (onedate format)

---

## Pipeline Steps

The end-to-end pipeline executes the following steps:

1. **Data Annotation** (`data_annotator.py`)
   - Generates structured annotations from clinical text using the LLM
   - Applies few-shot learning examples (for foundation model)
   - Outputs JSON with extracted entities and relationships

2. **PSSC Label Inference** (`infer_pssc_labelV3.py`)
   - Maps extracted field values to standardized PSSC labels
   - Uses sentence embeddings for similarity matching
   - Applies configurable thresholds for matching

3. **Biomarker Inference** (`infer_biomarkersname.py`)
   - Infers biomarker names from context
   - Filters measurement data
   - Maps to standardized biomarker vocabulary

4. **Remove "Other" Entries** (`postprocess_remove_other.py`)
   - Removes entries with generic "other" values
   - Cleans up low-information annotations

5. **Context Alignment** (`postprocess_replace_context_sentence.py`)
   - Aligns context sentences with original input text
   - Ensures sentence-level traceability

6. **Patient-level Merging** (`merge_predictions_by_patient.py`)
   - Merges document-level annotations by patient
   - Deduplicates entries based on key fields and date overlap
   - Applies biomarker mapping transformations
   - Filters redundant measurements and molecules
   - Converts date ranges to single dates

---

## Configuration

### Config Files

- **`conf/config_bigmodel.yaml`**: Foundation model configuration (medgemma-27b)
- **`conf/config_fastmodel.yaml`**: Fine-tuned model configuration (medgemma-4b)

### Key Configuration Options

```bash
# Process only a subset of files
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    num_samples=10

# Use GPU for embedding models
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    pssc_labels.device=cuda \
    pssc_labels.batch_size=256

# Disable specific pipeline steps
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    remove_other.enabled=false \
    merge_by_patient.enabled=false

# Keep temporary files for debugging
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    keep_temp=true

# Override vLLM server settings
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    data_annotator.vllm_base_url=http://localhost:8001/v1 \
    data_annotator.max_workers=32
```

---

## Troubleshooting

### vLLM server not responding

Ensure the vLLM server is running and accessible:
```bash
curl http://localhost:8000/v1/models
```

### CUDA out of memory

Reduce batch size in the configuration:
```bash
python inference_end_to_end.py \
    pssc_labels.batch_size=64 \
    ...
```

### Missing dependencies

Ensure the virtual environment is activated:
```bash
source .venv/bin/activate
```

### Pipeline errors

Enable debug mode by keeping temporary files:
```bash
python inference_end_to_end.py \
    keep_temp=true \
    ...
```

The temporary files will be preserved in `/tmp/inference_pipeline_*` for inspection.

---

## Notes

- The **foundation model** (Mode 1) provides higher quality but requires more GPU memory and processing time
- The **fine-tuned model** (Mode 2) is optimized for speed and uses domain-specific training
- Date ranges are automatically converted to single average dates in the final output
- Patient-level merging deduplicates entries based on overlapping date ranges and key field matching
- All text fields are UTF-8 encoded
