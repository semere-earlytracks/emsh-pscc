import json
import difflib
import sys
import enum
from datetime import date, datetime

from pydantic_schema import Document

sample = {
    "documentid": "doc-1",
    "personal_medical_history_comorbidities_and_adverse": [
        {
            "relatedpathologycode": "Acute renal failur",
            "relateddiagnosisdate": "12/03/2020",
            "contextsentence": "Patient had kidney failure."
        }
    ],
    "primary_tumor": [
        {
            "cancerdiagnosisdate": "01/02/2021",
            "topographycode": "C50 BREAST",
            "morphologycode": "85003, Infiltrating duct carcinoma",
            "tumorsize": []
        }
    ],
    "general_condition_and_physical_examination": [
        {
            "measuretype": "PS OMS",
            "measurevalue": "3",
            "measuredate_first": "10/10/2020"
        }
    ],
    "biomarkers_and_tumor_markers": [
        {
            "biomarkername": "CEA",
            "biomarkervaluetxt": "5 ng/mL",
            "biomarkerresultdate": "05/05/2020"
        }
    ],
    "tumor_events": [
        {
            "tumeventtype": "Local relapse",
            "tumeventdiagnosisdate": "03/03/2022",
            "metastasis": [
                {
                    "metastasistopocode": "C50, BREAST",
                    "metastasisdiscoverydate": "03/03/2022"
                }
            ]
        }
    ]
}

print("--- INPUT (raw) ---")
print(json.dumps(sample, indent=2, ensure_ascii=False))

try:
    doc = Document.model_validate(sample)
except Exception as e:
    print("Validation error:", e, file=sys.stderr)
    raise

print("\n--- PARSED (model_dump -> json-serializable) ---")
def _convert(value):
    # Convert Enum members and dates to JSON-serializable forms
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value

parsed = _convert(doc.model_dump())
print(json.dumps(parsed, indent=2, ensure_ascii=False))

print("\n--- UNIFIED DIFF (input -> parsed) ---")
left = json.dumps(sample, indent=2, ensure_ascii=False).splitlines()
right = json.dumps(parsed, indent=2, ensure_ascii=False).splitlines()
for line in difflib.unified_diff(left, right, fromfile='input', tofile='parsed', lineterm=''):
    print(line)

print("\nDone.")
