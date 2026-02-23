#!/usr/bin/env python3
"""Merge per-document prediction JSON files into a patient-level merged JSON.

Produces a JSON with structure:
{
    "patients": [ ... ],
    "documents": [ ... ]
}

- `documents`: concatenation of each document JSON plus a `documentid` field (filename without .json).
- `patients`: for each patient directory, concatenates list values found in document JSONs. Non-list scalar values are copied from the first occurrence.

Usage:
    python merge_predictions.py /path/to/prediction-input-dir merged_predictions.json
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict


def merge_dicts_concat(dst: Dict[str, Any], src: Dict[str, Any], documentid: str | None = None) -> None:
    """Recursively merge `src` into `dst`.

    - When both values are lists: extend `dst` with `src`.
    - When both are dicts: recurse.
    - When types differ and `src` is a list: replace `dst` with `src` (best-effort).
    - For scalar conflicts: keep existing `dst` value (first-seen wins).
    """
    for k, v in src.items():
        if k not in dst:
            # When introducing a new key, copy but annotate list dict-entries with documentid
            if isinstance(v, list):
                new_list = []
                for item in deepcopy(v):
                    if isinstance(item, dict) and documentid is not None:
                        item.setdefault("documentid", documentid)
                    new_list.append(item)
                dst[k] = new_list
            elif isinstance(v, dict):
                # create empty dict then merge to ensure nested lists get annotated
                dst[k] = {}
                merge_dicts_concat(dst[k], v, documentid)
            else:
                dst[k] = deepcopy(v)
            continue

        a = dst[k]
        b = v

        if isinstance(a, list) and isinstance(b, list):
            for item in deepcopy(b):
                if isinstance(item, dict) and documentid is not None:
                    item.setdefault("documentid", documentid)
                a.append(item)
        elif isinstance(a, dict) and isinstance(b, dict):
            merge_dicts_concat(a, b, documentid)
        elif isinstance(b, list):
            # src has a list but dst doesn't — use src list (annotated)
            new_list = []
            for item in deepcopy(b):
                if isinstance(item, dict) and documentid is not None:
                    item.setdefault("documentid", documentid)
                new_list.append(item)
            dst[k] = new_list
        else:
            # leave dst as-is for scalar or mismatched types
            continue


# ---------------------------------------------------------------------
# Merge-by-key with date-overlap utilities
# ---------------------------------------------------------------------

_KEY_PROPS = [
    "relatedpathologycode",
    "topographycode",
    "morphologycode",
    "measuretype",
    "surgerytype",
    "moleculecode",
    "radiotherapytype",
    "imagingmodality",
    "specimentype",
    "specimennature",
    "specimentopographycode",
    "biomarkername",
    "metastasistopocode",
    "tumeventtype",
]


def _get_entry_key(entry: Any) -> str | None:
    """Return a string key for an entry.

    - If `entry` is not a dict, return its string value.
    - If `entry` is a dict, concatenate any of the properties in `_KEY_PROPS`
      that are present (in the listed order) separated by '|'. If none
      of those properties are present, return None.
    """
    if not isinstance(entry, dict):
        try:
            return str(entry).strip().lower()
        except Exception:
            return None

    parts = []
    for prop in _KEY_PROPS:
        if prop in entry and entry[prop] is not None:
            val = entry[prop]
            if isinstance(val, list):
                normalized = "|".join(str(x).strip().lower() for x in val)
                parts.append(normalized)
            else:
                parts.append(str(val).strip().lower())
    if parts:
        return "|".join(parts)
    return None


def _find_date_range(obj: Any) -> tuple[date, date] | None:
    """Recursively find the first date-range object for a given entry.

    Looks for dict keys containing 'date' whose value is a dict with
    'start' and 'end' ISO date strings. Returns (start_date, end_date)
    as `date` objects, or `None` if not found.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "date" in k.lower() and isinstance(v, dict) and "start" in v and "end" in v:
                try:
                    s = date.fromisoformat(v["start"])
                    e = date.fromisoformat(v["end"])
                    return (s, e)
                except Exception:
                    pass
            # recurse
            sub = _find_date_range(v)
            if sub:
                return sub
    elif isinstance(obj, list):
        for item in obj:
            sub = _find_date_range(item)
            if sub:
                return sub
    return None


def _ranges_overlap(a: tuple[date, date], b: tuple[date, date]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _duration_days(r: tuple[date, date]) -> int:
    return (r[1] - r[0]).days


def _set_all_date_ranges(obj: Any, start: date, end: date) -> None:
    """Recursively set all found date-range dicts to the given start/end (ISO strings)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "date" in k.lower() and isinstance(v, dict) and "start" in v and "end" in v:
                v["start"] = start.isoformat()
                v["end"] = end.isoformat()
            else:
                _set_all_date_ranges(v, start, end)
    elif isinstance(obj, list):
        for item in obj:
            _set_all_date_ranges(item, start, end)


def merge_entries_by_key(entries: list[dict]) -> list[dict]:
    """Merge entries that share the same key and have overlapping date ranges.

    Rules:
    - Compute a key for each entry using `_get_entry_key`.
    - Only entries with the same non-None key are considered for merging.
    - An entry must have a date-range (found via `_find_date_range`) to be
      mergeable. Two entries are merged only if their ranges overlap.
    - When merging two entries, the resulting entry copies everything from
      the entry with the smaller (more specific) time range; in case of a tie
      the first entry wins. The date-range(s) in the resulting entry are set
      to the intersection (overlap) of the two ranges.

    This function performs transitive merging: any entries in a group that
    are connected by overlapping ranges will be merged into a single entry.
    """
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    others: list[dict] = []

    for ent in entries:
        key = _get_entry_key(ent)
        if key is None:
            others.append(ent)
        else:
            # If the concatenated key is exactly "other" (case-sensitive), drop it
            if key == "other":
                continue
            buckets[key].append(ent)

    out: list[dict] = []

    for key, group in buckets.items():
        # Work on a mutable list copy
        remaining = list(group)
        while remaining:
            current = remaining.pop(0)
            cur_range = _find_date_range(current)
            if cur_range is None:
                # cannot merge by date if current has no range; just keep it
                out.append(current)
                continue

            i = 0
            while i < len(remaining):
                candidate = remaining[i]
                cand_range = _find_date_range(candidate)
                if cand_range is None:
                    i += 1
                    continue

                if _ranges_overlap(cur_range, cand_range):
                    # choose base = entry with smaller duration (more specific)
                    if _duration_days(cand_range) < _duration_days(cur_range):
                        base = deepcopy(candidate)
                    else:
                        base = deepcopy(current)

                    # compute overlap
                    new_start = max(cur_range[0], cand_range[0])
                    new_end = min(cur_range[1], cand_range[1])

                    # set all date ranges in base to overlap
                    _set_all_date_ranges(base, new_start, new_end)

                    # continue merging: new current is base; remove candidate
                    current = base
                    cur_range = (new_start, new_end)
                    remaining.pop(i)
                    # restart scanning from beginning (transitive merges)
                    i = 0
                else:
                    i += 1

            out.append(current)

    # append non-keyed entries at the end (unchanged)
    out.extend(others)

    # Sort by average of start+end date (midpoint). Entries without a date
    # range are placed after ranged entries.
    def _avg_midpoint(ent: dict) -> float:
        r = _find_date_range(ent)
        if r is None:
            return float("inf")
        # use ordinal average of dates for stable numeric sorting
        return (r[0].toordinal() + r[1].toordinal()) / 2.0

    out.sort(key=_avg_midpoint)
    return out


def _apply_merge_to_lists(obj: Any) -> None:
    """Recursively walk `obj` and replace any lists with their merged form.

    For dict values that are lists, call `merge_entries_by_key` and set the
    result back into the dict. Also recurse into nested dicts and list
    elements to handle nested lists (e.g., lists inside 'patient_history').
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, list):
                try:
                    obj[k] = merge_entries_by_key(v)
                except Exception:
                    # If merge fails for this list, leave it unchanged but
                    # still recurse into the elements for nested structures.
                    for item in v:
                        _apply_merge_to_lists(item)
            else:
                _apply_merge_to_lists(v)
    elif isinstance(obj, list):
        for item in obj:
            _apply_merge_to_lists(item)


def _apply_biomarker_mapping(obj: Any) -> None:
    """Recursively update biomarker value text using mutation/non-mutation fields.

    For dicts containing `biomarkermutationstatus` (M),
    `biomarkernonmutationstatus` (NM) and `biomarkervaluetxt` (TXT), set
    `biomarkervaluetxt` to:
        f"{M} / {NM} ({TXT})" if M != "Other"
    otherwise:
        f"{NM} ({TXT})"

    If any piece is missing, it's skipped gracefully; TXT is omitted if empty.
    The function mutates `obj` in-place.
    """
    if isinstance(obj, dict):
        # If this dict looks like a biomarker entry, transform it
        if "biomarkermutationstatus" in obj:
            M = obj.get("biomarkermutationstatus")
            NM = obj.get("biomarkernonmutationstatus")
            TXT = obj.get("biomarkervaluetxt")
            m_str = str(M).strip() if M is not None else ""
            nm_str = str(NM).strip() if NM is not None else ""
            txt_str = str(TXT).strip() if TXT is not None else ""

            if m_str and m_str != "Other":
                if nm_str:
                    main = f"{m_str} / {nm_str}"
                else:
                    main = m_str
            else:
                main = nm_str

            if main:
                if txt_str:
                    obj["biomarkervaluetxt"] = f"{main} ({txt_str})"
                else:
                    obj["biomarkervaluetxt"] = main
                
                if "biomarkermutationstatus" in obj:
                    del obj["biomarkermutationstatus"]
                if "biomarkernonmutationstatus" in obj:
                    del obj["biomarkernonmutationstatus"]

        # Recurse into children
        for k, v in obj.items():
            _apply_biomarker_mapping(v)
    elif isinstance(obj, list):
        for item in obj:
            _apply_biomarker_mapping(item)


def process_patient_dir(patient_dir: Path, documents: list, patients_out: list) -> None:
    """Process one patient directory: append documents and build merged patient entry."""
    merged_patient: Dict[str, Any] = {"patientid": patient_dir.name}

    for p in sorted(patient_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".json":
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")
            continue

        # If the JSON has a top-level "output" key, use its value as the document data (to handle wrapped outputs);
        if "output" in data and isinstance(data["output"], dict):
            data = data["output"]

        # Add document entry (copy of file JSON with documentid added)
        doc_entry = deepcopy(data)
        doc_entry["documentid"] = p.stem
        documents.append(doc_entry)

        # Decide what to merge into patient:
        # Prefer an explicit top-level `patient` object if present and is a dict,
        # otherwise merge the whole document JSON.
        if isinstance(data.get("patient"), dict):
            source = data["patient"]
        else:
            source = data

        merge_dicts_concat(merged_patient, source, documentid=p.stem)

    # After concatenating lists, attempt to merge list entries by key+date
    # overlap and sort them by midpoint.
    _apply_merge_to_lists(merged_patient)

    # Apply biomarker mapping to normalize biomarker text fields
    _apply_biomarker_mapping(merged_patient)

    patients_out.append(merged_patient)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge per-document prediction JSONs into patients+documents JSON.")
    p.add_argument("input_dir", help="Directory containing patient subdirectories with per-document JSON files")
    p.add_argument("output", nargs="?", default="merged_predictions.json", help="Output JSON path")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    documents = []
    patients = []

    # Iterate patient subdirectories (one patient per subdirectory)
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            process_patient_dir(child, documents, patients)

    out = {"patients": patients, "documents": documents}

    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote merged output to {out_path}")

    # Also produce a version where all {"start","end"} ranges are replaced
    # by their average date in dd/mm/YYYY format. If the averaged date is
    # earlier than 01/01/1900, clamp to that date.
    def _average_date_str(start_iso: str, end_iso: str) -> str:
        try:
            s = date.fromisoformat(start_iso)
            e = date.fromisoformat(end_iso)
        except Exception:
            return "01/01/1900"
        avg_ord = round((s.toordinal() + e.toordinal()) / 2.0)
        avg = date.fromordinal(int(avg_ord))
        min_date = date(1900, 1, 1)
        if avg < min_date:
            avg = min_date
        return avg.strftime("%d/%m/%Y")

    def _convert_ranges_to_onedate(obj: Any) -> Any:
        # Returns a transformed copy of obj where any dict with 'start' and
        # 'end' keys is replaced with the averaged date string.
        if isinstance(obj, dict):
            # Detect a plain range dict
            if set(obj.keys()) >= {"start", "end"} and len(obj.keys()) == 2:
                return _average_date_str(obj.get("start", ""), obj.get("end", ""))
            new = {}
            for k, v in obj.items():
                new[k] = _convert_ranges_to_onedate(v)
            return new
        elif isinstance(obj, list):
            return [_convert_ranges_to_onedate(i) for i in obj]
        else:
            return obj

    # Build output path for the one-date file: insert '.onedate' before .json
    if out_path.suffix == ".json":
        one_path = out_path.with_name(out_path.stem + ".onedate.json")
    else:
        one_path = Path(str(out_path) + ".onedate.json")

    one_obj = _convert_ranges_to_onedate(deepcopy(out))
    one_path.write_text(json.dumps(one_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote one-date output to {one_path}")


if __name__ == "__main__":
    main()
