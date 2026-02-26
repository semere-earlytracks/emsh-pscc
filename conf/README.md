# Configuration for inference_end_to_end.py

This directory contains the Hydra configuration files for the end-to-end inference pipeline.

## Configuration Files

- **config_bigmodel.yaml**: Configuration for large foundation model (e.g., medgemma-27b) with few-shot examples
- **config_fastmodel.yaml**: Configuration for fine-tuned model (checkpoint-300) without few-shot examples

## Model Configurations

### Big Model (config_bigmodel.yaml)
- Uses large foundation model: `ig1/medgemma-27b-text-it-FP8-Dynamic`
- Enables few-shot examples (`disable_few_shot: false`)
- Best for: High accuracy, zero-shot scenarios, initial model evaluation

### Fast Model (config_fastmodel.yaml)
- Uses fine-tuned model: `model/checkpoint-300`
- Disables few-shot examples (`disable_few_shot: true`)
- Best for: Production deployment, faster inference, fine-tuned models

## Pipeline Steps Configuration

The configuration files are organized by pipeline steps:

### 1. Data Annotator (vLLM annotation generation)
```yaml
data_annotator:
  vllm_base_url: http://localhost:8000/v1
  vllm_api_key: EMPTY
  model_name: ig1/medgemma-27b-text-it-FP8-Dynamic
  max_workers: 64
  temperature: 0.0
  max_tokens: 8192
  disable_few_shot: false  # Set to true to disable few-shot examples
  use_response_format: true  # Use structured output (set false for fine-tuned models)
```

### 2. PSSC Label Inference (embedding-based mapping)
```yaml
pssc_labels:
  model: all-MiniLM-L6-v2
  labels_dir: data/pssc-labels
  batch_size: 128
  device: null  # auto-detect
  strip_codes: false  # keep full labels
```

### 3. Biomarker Inference (exact string matching)
```yaml
biomarkers:
  labels_csv: data/pssc-labels/biomarkername_ext.csv
  case_sensitive: false
```

### 4-6. Postprocessing Steps
```yaml
remove_other:
  enabled: true
replace_context:
  enabled: true
merge_by_patient:
  enabled: true
```

## Usage Examples

### Using Big Model (Default)
```bash
# Basic usage - uses config_bigmodel.yaml by default
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output

# With GPU acceleration
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    pssc_labels.device=cuda \
    pssc_labels.batch_size=256
```

### Using Fast Model
```bash
# Use fine-tuned model without few-shot examples
python inference_end_to_end.py \
    --config-name=config_fastmodel \
    input_dir=data/input \
    output_dir=data/output

# With higher parallelism for fast model
python inference_end_to_end.py \
    --config-name=config_fastmodel \
    input_dir=data/input \
    output_dir=data/output \
    data_annotator.max_workers=512
```

### Processing Subset
```bash
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    num_samples=10
```

### Disable Specific Steps
```bash
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    remove_other.enabled=false \
    merge_by_patient.enabled=false
```

### Custom vLLM Settings
```bash
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    data_annotator.vllm_base_url=http://server:8001/v1 \
    data_annotator.max_workers=32
```

### Debug Mode
```bash
python inference_end_to_end.py \
    input_dir=data/input \
    output_dir=data/output \
    num_samples=5 \
    keep_temp=true
```

## Environment Variables

The following environment variables can be used (they override config values):
- `VLLM_BASE_URL`: vLLM server URL
- `VLLM_API_KEY`: vLLM API key
- `VLLM_MODEL_NAME`: Model name for vLLM
- `DISABLE_FEW_SHOT`: Set to "true" or "1" to disable few-shot examples
- `USE_RESPONSE_FORMAT`: Set to "true" or "1" to use structured output (default: true)
- `MAX_WORKERS`: Number of parallel workers for vLLM
- `TEMPERATURE`: Sampling temperature
- `MAX_TOKENS`: Maximum tokens per generation

## Output

The pipeline produces:
- **Intermediate temporary directories** (deleted unless `keep_temp=true`):
  - step1_annotated: Raw vLLM annotations
  - step2_pssc_labels: After label inference
  - step3_biomarkers: After biomarker inference
  - step4_no_other: After removing "other" entries
  - step5_context_aligned: After context sentence alignment

- **Final output**:
  - If `merge_by_patient.enabled=true`: A single `predictions.json` file in `output_dir`
  - If `merge_by_patient.enabled=false`: Individual JSON files preserving directory structure

## Customizing Configuration

You can:
1. **Choose a config file**: Use `--config-name=config_bigmodel` (default) or `--config-name=config_fastmodel`
2. **Edit config files**: Modify `config_bigmodel.yaml` or `config_fastmodel.yaml` to change default values
3. **Override on command line**: Use Hydra's override syntax for any parameter

Example overrides:
```bash
python inference_end_to_end.py \
    --config-name=config_fastmodel \
    input_dir=data/input \
    output_dir=data/output \
    pssc_labels.model=sentence-transformers/all-mpnet-base-v2 \
    biomarkers.case_sensitive=true \
    final_merged_json=my_custom_output.json
```

For more information on Hydra configuration, see [Hydra documentation](https://hydra.cc/).
