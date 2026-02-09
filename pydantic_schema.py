from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Union, Literal, Annotated
from typing import get_origin, get_args
import csv
from pathlib import Path
import enum
import difflib
import sys

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_eu_date(value: str | None) -> date | None:
    """Parse European date format DD/MM/YYYY into datetime.date."""
    if value is None:
        return None
    try: 
        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%d-%m-%Y").date()
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


class BaseDocumentModel(BaseModel):
    """
    Base model for all document-level entities.

    - Extra fields are allowed to avoid over-constraining LLM output.
    - Field descriptions are short and human/LLM readable.
    """
    model_config = {
        "extra": "allow",
        "populate_by_name": True,
    }
    
    def to_json_serializable(self) -> dict:
        """Return a JSON-serializable representation of the model.

        This recursively converts `date` and `datetime` objects to ISO-formatted
        strings so the result can be dumped with the standard `json` module.
        """
        from datetime import date, datetime

        def _convert(value):
            if isinstance(value, enum.Enum):
                return value.value
            if isinstance(value, (date, datetime)):
                return value.isoformat()
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_convert(v) for v in value]
            return value

        return _convert(self.model_dump())

    @model_validator(mode="before")
    @classmethod
    def _ensure_contextsentence_present(cls, values):
        # If the model declares a `contextsentence` field but the input omits it,
        # inject an empty string so required validation passes and downstream
        # coercion can normalize None -> "" as needed.
        try:
            if isinstance(values, dict) and "contextsentence" in cls.model_fields:
                if "contextsentence" not in values:
                    values["contextsentence"] = ""
        except Exception:
            pass
        return values


# ---------------------------------------------------------------------
# Load label CSV and build enums for Literal-like fields
# ---------------------------------------------------------------------
def _load_pscc_labels() -> dict:
    """Load pscc_labels.csv from project root (if present) and return mapping of column->list of values."""
    root = Path(__file__).resolve().parent
    # Look for csv in repo root or current directory
    candidates = [root / "pscc_labels.csv", root.parent / "pscc_labels.csv", Path("pscc_labels.csv")]
    for p in candidates:
        if p.exists():
            # Increase CSV field size limit to handle very large fields
            try:
                csv.field_size_limit(sys.maxsize)
            except OverflowError:
                csv.field_size_limit(10 * 1024 * 1024)

            with p.open("r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                labels = {}
                for row in reader:
                    if not row:
                        continue
                    # Expect first column = property name, second column = pipe-separated labels
                    key = row[0].strip().lower()
                    if not key:
                        continue
                    # Combine all remaining columns into one string in case labels contain commas
                    value = "|".join(row[1:]) if len(row) > 1 else ""
                    if value is None:
                        continue
                    # Normalize and split on pipe
                    items = [item.strip() for item in str(value).split("|") if item.strip()]
                    if items:
                        labels.setdefault(key, []).extend(items)
                return labels
    raise FileNotFoundError("pscc_labels.csv not found in project root or current directory")


def _make_literal_type(name: str, values: List[str], default: List[str]):
    vals = tuple(values or default)
    try:
        return Literal.__getitem__(vals)
    except Exception:
        return str


_PSCC = _load_pscc_labels()


# Provide Literal type aliases (fallbacks) for static type checkers.
# At runtime these names will be overwritten with Enum classes when the CSV is loaded.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    RelatedPathologyCode = Literal[
        "N17, Acute renal failure",
        "K72, Hepatic failure",
        "I50, Heart failure",
        "E11, Type 2 diabetes mellitus",
        "J45, Asthma",
        "I21, Acute myocardial infarction",
        "E66, Obesity",
    ]
    TopographyCode = Literal[
        "C50, BREAST",
        "C04, FLOOR OF MOUTH",
        "C18, COLON",
        "C34, BRONCHUS AND LUNG",
        "C20, RECTUM",
        "C16, STOMACH",
        "C25, PANCREAS",
        "C67, BLADDER",
    ]
    MorphologyCode = Literal[
        "80102, Carcinoma in situ",
        "81403, Adenocarcinoma",
        "85003, Infiltrating duct carcinoma",
        "80703, Squamous cell carcinoma",
        "89603, Malignant neoplasm, NOS",
    ]
    MeasureType = Literal[
        "PS OMS",
        "PS Karnofsky",
        "Height in cms",
        "Weight",
        "BMI",
        "Blood pressure",
        "Heart rate",
        "Temperature",
    ]
    SurgeryType = Literal[
        "06.03.02, Lobectomie pulmonaire",
        "06.03.01, Segmentectomy",
        "06.03.03, Pneumonectomy",
        "01.02.01, Partial hepatectomy",
        "02.05.04, Radical mastectomy",
    ]
    MoleculeCode = Literal[
        "L01XC18, Pembrolizumab",
        "L01AA01, Cyclophosphamide",
        "L01BC01, Doxorubicin",
        "L01XA02, Trastuzumab",
        "L01XA03, Bevacizumab",
    ]
    RadiotherapyType = Literal[
        "Total body irradiation",
        "External beam radiotherapy",
        "Brachytherapy",
        "Stereotactic radiotherapy",
        "Proton therapy",
    ]
    ImagingModality = Literal[
        "CT, Computed Tomography",
        "MR, Magnetic Resonance",
        "XR, X-ray",
        "US, Ultrasound",
        "PET, Positron Emission Tomography",
        "NM, Nuclear Medicine",
    ]
    SpecimenType = Literal[
        "Biopsy",
        "Surgical resection",
        "Fine needle aspiration",
        "Cytology",
        "Blood sample",
    ]
    SpecimenNature = Literal[
        "Tumor tissue",
        "Normal tissue",
        "Adjacent tissue",
        "Lymph node",
        "Blood",
    ]
    SpecimenTopographyCode = Literal[
        "C04, FLOOR OF MOUTH",
        "C50, BREAST",
        "C34, BRONCHUS AND LUNG",
        "C18, COLON",
        "C67, BLADDER",
    ]
    BiomarkerName = Literal[
        "ACE",
        "CEA",
        "CA19-9",
        "AFP",
        "PSA",
    ]
    MetastasisTopocode = Literal[
        "C40 BONES, JOINTS AND ARTICULAR CARTILAGE OF LIMBS",
        "C50, BREAST",
        "C34, BRONCHUS AND LUNG",
        "C77, LYMPH NODES",
        "C18, COLON",
    ]
    TumeventType = Literal[
        "Local/regional relapse or Recurrence",
        "Metastasis",
        "Recurrence",
        "Local relapse",
        "Distant metastasis",
        "Progression",
    ]
else:
    # runtime placeholders; will be replaced by Enum classes below
    RelatedPathologyCode = str
    TopographyCode = str
    MorphologyCode = str
    MeasureType = str
    SurgeryType = str
    MoleculeCode = str
    RadiotherapyType = str
    ImagingModality = str
    SpecimenType = str
    SpecimenNature = str
    SpecimenTopographyCode = str
    BiomarkerName = str
    MetastasisTopocode = str
    TumeventType = str


# Create Enum classes for all keys present in the CSV or fallbacks.
_ENUMS = {}
# Explicit PascalCase names for the enums we want to expose
_NAME_MAP = {
    "relatedpathologycode": "RelatedPathologyCode",
    "topographycode": "TopographyCode",
    "morphologycode": "MorphologyCode",
    "measuretype": "MeasureType",
    "surgerytype": "SurgeryType",
    "moleculecode": "MoleculeCode",
    "radiotherapytype": "RadiotherapyType",
    "imagingmodality": "ImagingModality",
    "specimentype": "SpecimenType",
    "specimennature": "SpecimenNature",
    "specimentopographycode": "SpecimenTopographyCode",
    "biomarkername": "BiomarkerName",
    "metastasistopocode": "MetastasisTopocode",
    "tumeventtype": "TumeventType",
}

for key in _NAME_MAP.keys():
    values = _PSCC.get(key)
    type_name = _NAME_MAP.get(key)
    if values:
        # Create a Literal[...] runtime type from CSV values
        _ENUMS[key] = _make_literal_type(type_name, values, [])
        globals()[type_name] = _ENUMS[key]
    else:
        # Keep runtime placeholder (str) and warn the user
        print(f"Warning: pscc_labels.csv does not contain column '{key}'; using 'str' for {type_name}", file=sys.stderr)
        globals()[type_name] = globals().get(type_name, str)


# ---------------------------------------------------------------------
# Helper: coerce values to Enum members using fuzzy matching
# ---------------------------------------------------------------------
def _coerce_enum_value(enum_cls, v):
    if v is None:
        return v

    s = str(v).strip()
    if not s:
        return v

    # If enum.Enum subclass, handle as before and return an Enum member
    try:
        if isinstance(enum_cls, type) and issubclass(enum_cls, enum.Enum):
            # direct match
            for m in enum_cls:
                if s == m.value or s.lower() == str(m.value).lower():
                    return m
            # fuzzy
            choices = [str(m.value) for m in enum_cls]
            matches = difflib.get_close_matches(s, choices, n=1, cutoff=0.5)
            if matches:
                match = matches[0]
                for m in enum_cls:
                    if str(m.value) == match:
                        return m
            return list(enum_cls)[0]
    except Exception:
        pass

    # If a typing.Literal[...] runtime object, coerce to the matching string
    try:
        if get_origin(enum_cls) is Literal:
            choices = list(get_args(enum_cls))
            # direct match
            for c in choices:
                if s == c or s.lower() == str(c).lower():
                    return c
            # fuzzy
            matches = difflib.get_close_matches(s, [str(c) for c in choices], n=1, cutoff=0.5)
            if matches:
                return matches[0]
            # fallback to first choice
            if choices:
                return choices[0]
    except Exception:
        pass

    return v


# ---------------------------------------------------------------------
# Personal medical history
# ---------------------------------------------------------------------

class PersonalMedicalHistory(BaseDocumentModel):
    """Comorbidity or adverse medical history item."""

    relatedpathologycode: RelatedPathologyCode = Field(..., description="Pathology or condition code")

    relateddiagnosisdate: Optional[date] = Field(
        None, description="Date of diagnosis"
    )
    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("relateddiagnosisdate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)
    @field_validator("contextsentence", mode="before")
    @classmethod
    def _coerce_contextsentence(cls, v):
        if v is None:
            return ""
        return v

    @field_validator("relatedpathologycode", mode="before")
    @classmethod
    def _coerce_relatedpathologycode(cls, v):
        enum_cls = globals().get("RelatedPathologyCode")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Tumor size (discriminated union)
# ---------------------------------------------------------------------

class TumorSizeBase(BaseDocumentModel):
    """Base class for a tumor size measurement."""
    kind: str = Field(..., description="Discriminator for tumor size type")

    contextsentence: str = Field(..., description="Sentence supporting this information")


class TumorSizeClinic(TumorSizeBase):
    """Clinical tumor size."""
    kind: Literal["clinic"] = Field("clinic", description="Clinical measurement")

    tumorsize_clinic: str = Field(..., description="Clinical tumor size value")
    tumorsizedate_clinic: date = Field(
        ..., description="Date of clinical measurement"
    )

    @field_validator("tumorsizedate_clinic", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("contextsentence", mode="before")
    @classmethod
    def _coerce_contextsentence(cls, v):
        if v is None:
            return ""
        return v


class TumorSizePatho(TumorSizeBase):
    """Pathological tumor size."""
    kind: Literal["patho"] = Field("patho", description="Pathological measurement")

    tumorsize_patho: str = Field(..., description="Pathological tumor size value")
    tumorsizedate_patho: date = Field(
        ..., description="Date of pathological measurement"
    )

    @field_validator("tumorsizedate_patho", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)


class TumorSizeImaging(TumorSizeBase):
    """Imaging-based tumor size."""
    kind: Literal["imaging"] = Field("imaging", description="Imaging measurement")

    tumorsize_imaging: str = Field(..., description="Imaging tumor size value")
    tumorsizedate_imaging: date = Field(
        ..., description="Date of imaging measurement"
    )

    @field_validator("tumorsizedate_imaging", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)


TumorSize = Annotated[
    Union[TumorSizeClinic, TumorSizePatho, TumorSizeImaging],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------
# Primary tumor
# ---------------------------------------------------------------------

class PrimaryTumor(BaseDocumentModel):
    """Primary tumor description."""

    cancerdiagnosisdate: date = Field(
        ..., description="Date of cancer diagnosis"
    )

    topographycode: TopographyCode = Field(..., description="Tumor topography code")

    morphologycode: MorphologyCode = Field(..., description="Tumor morphology code")

    tumorsize: List[TumorSize] = Field(
        default_factory=list,
        description="Tumor size measurements"
    )

    @field_validator("cancerdiagnosisdate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("topographycode", mode="before")
    @classmethod
    def _coerce_topographycode(cls, v):
        enum_cls = globals().get("TopographyCode")
        return _coerce_enum_value(enum_cls, v)

    @field_validator("morphologycode", mode="before")
    @classmethod
    def _coerce_morphologycode(cls, v):
        enum_cls = globals().get("MorphologyCode")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# General condition & physical examination
# ---------------------------------------------------------------------

class GeneralCondition(BaseDocumentModel):
    """Functional status or physical measurement."""

    measuretype: MeasureType = Field(..., description="Type of measurement")

    measurevalue: str = Field(..., description="Measured value")

    measuredate_first: date = Field(
        ..., description="Date of measurement"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("measuredate_first", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("measuretype", mode="before")
    @classmethod
    def _coerce_measuretype(cls, v):
        enum_cls = globals().get("MeasureType")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Surgery
# ---------------------------------------------------------------------

class Surgery(BaseDocumentModel):
    """Surgical procedure."""

    surgerytype: SurgeryType = Field(..., description="Surgery type or code")

    surgerydate: date = Field(..., description="Date of surgery")

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("surgerydate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("surgerytype", mode="before")
    @classmethod
    def _coerce_surgerytype(cls, v):
        enum_cls = globals().get("SurgeryType")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Cancer medication
# ---------------------------------------------------------------------

class CancerMedication(BaseDocumentModel):
    """Systemic anticancer treatment."""

    moleculecode: MoleculeCode = Field(..., description="Drug or molecule code")

    moleculedate_first: date = Field(
        ..., description="First administration date"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("moleculedate_first", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("moleculecode", mode="before")
    @classmethod
    def _coerce_moleculecode(cls, v):
        enum_cls = globals().get("MoleculeCode")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Radiotherapy
# ---------------------------------------------------------------------

class Radiotherapy(BaseDocumentModel):
    """Radiotherapy treatment."""

    radiotherapytype: RadiotherapyType = Field(..., description="Type of radiotherapy")

    radiotherapydate_first: date = Field(
        ..., description="First radiotherapy date"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("radiotherapydate_first", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("radiotherapytype", mode="before")
    @classmethod
    def _coerce_radiotherapytype(cls, v):
        enum_cls = globals().get("RadiotherapyType")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Progression
# ---------------------------------------------------------------------

class Progression(BaseDocumentModel):
    """Disease progression event."""

    progressiondate: date = Field(
        ..., description="Date of progression"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("progressiondate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)


# ---------------------------------------------------------------------
# Imaging
# ---------------------------------------------------------------------

class Imaging(BaseDocumentModel):
    """Imaging or nuclear medicine exam."""

    imagingmodality: ImagingModality = Field(..., description="Imaging modality")

    analysisdate: date = Field(
        ..., description="Date of imaging"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("analysisdate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("imagingmodality", mode="before")
    @classmethod
    def _coerce_imagingmodality(cls, v):
        enum_cls = globals().get("ImagingModality")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Biological specimen
# ---------------------------------------------------------------------

class BiologicalSpecimen(BaseDocumentModel):
    """Collected biological specimen."""

    specimentype: SpecimenType = Field(
        ..., description="Type of specimen"
    )

    specimennature: SpecimenNature = Field(
        ..., description="Nature of specimen"
    )

    specimentopographycode: SpecimenTopographyCode = Field(..., description="Specimen topography code")

    specimencollectdateday: date = Field(
        ..., description="Specimen collection date"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("specimencollectdateday", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("specimentype", mode="before")
    @classmethod
    def _coerce_specimentype(cls, v):
        enum_cls = globals().get("SpecimenType")
        return _coerce_enum_value(enum_cls, v)

    @field_validator("specimennature", mode="before")
    @classmethod
    def _coerce_specimennature(cls, v):
        enum_cls = globals().get("SpecimenNature")
        return _coerce_enum_value(enum_cls, v)

    @field_validator("specimentopographycode", mode="before")
    @classmethod
    def _coerce_specimentopographycode(cls, v):
        enum_cls = globals().get("SpecimenTopographyCode")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Biomarkers
# ---------------------------------------------------------------------

class Biomarker(BaseDocumentModel):
    """Biomarker or tumor marker measurement."""

    biomarkername: BiomarkerName = Field(
        ..., description="Biomarker name"
    )

    biomarkervaluetxt: str = Field(
        ..., description="Biomarker value with unit"
    )

    biomarkerresultdate: date = Field(
        ..., description="Date of biomarker result"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("biomarkerresultdate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("biomarkername", mode="before")
    @classmethod
    def _coerce_biomarkername(cls, v):
        enum_cls = globals().get("BiomarkerName")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Metastasis & tumor events
# ---------------------------------------------------------------------

class Metastasis(BaseDocumentModel):
    """Metastatic lesion."""

    metastasistopocode: MetastasisTopocode = Field(..., description="Metastasis topography code")

    metastasisdiscoverydate: date = Field(
        ..., description="Date of metastasis discovery"
    )

    contextsentence: str = Field(..., description="Sentence supporting this information")

    @field_validator("metastasisdiscoverydate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("metastasistopocode", mode="before")
    @classmethod
    def _coerce_metastasistopocode(cls, v):
        enum_cls = globals().get("MetastasisTopocode")
        return _coerce_enum_value(enum_cls, v)


class TumorEvent(BaseDocumentModel):
    """Tumor-related clinical event."""

    tumeventtype: TumeventType = Field(..., description="Type of tumor event")

    tumeventdiagnosisdate: date = Field(
        ..., description="Date of event diagnosis"
    )

    metastasis: List[Metastasis] = Field(
        default_factory=list,
        description="Associated metastases"
    )

    @field_validator("tumeventdiagnosisdate", mode="before")
    @classmethod
    def _parse_date(cls, v):
        return parse_eu_date(v)

    @field_validator("tumeventtype", mode="before")
    @classmethod
    def _coerce_tumeventtype(cls, v):
        enum_cls = globals().get("TumeventType")
        return _coerce_enum_value(enum_cls, v)


# ---------------------------------------------------------------------
# Document (top-level)
# ---------------------------------------------------------------------

class Document(BaseDocumentModel):
    """Single extracted medical document."""

    documentid: str = Field(
        ..., description="Unique document identifier"
    )

    personal_medical_history_comorbidities_and_adverse: List[
        PersonalMedicalHistory
    ] = Field(default_factory=list)

    primary_tumor: List[PrimaryTumor] = Field(default_factory=list)

    general_condition_and_physical_examination: List[
        GeneralCondition
    ] = Field(default_factory=list)

    surgery: List[Surgery] = Field(default_factory=list)

    cancer_medication: List[CancerMedication] = Field(default_factory=list)

    radiotherapy: List[Radiotherapy] = Field(default_factory=list)

    progression: List[Progression] = Field(default_factory=list)

    imaging_and_nuclear_medecine: List[Imaging] = Field(default_factory=list)

    biological_specimen: List[BiologicalSpecimen] = Field(default_factory=list)

    biomarkers_and_tumor_markers: List[Biomarker] = Field(default_factory=list)

    tumor_events: List[TumorEvent] = Field(default_factory=list)
