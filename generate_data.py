"""Generate single Document+text examples using OpenAI and save them.

Usage:
  Set `OPENAI_API_KEY` in the environment. Optionally set `OPENAI_MODEL`.
  Run: `python generate_data.py`

This writes one JSON file into `data/generated/<md5>.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import openai
from pydantic import BaseModel

from pydantic_schema import Document


class DocumentAndText(BaseModel):
	date_of_document: str
	summary_of_case_till_now: Optional[str] = None
	document: Document
	text: str


def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
	# Prefer explicit api_key param, then env var
	key = api_key or os.environ.get("OPENAI_API_KEY")
	if not key:
		raise RuntimeError("OPENAI_API_KEY must be provided via --api-key or environment")

	# Allow overriding the API base (for vLLM / proxy setups)
	base = base_url or os.environ.get("OPENAI_BASE_URL") or None
	return openai.OpenAI(api_key=key, base_url=base)


def build_prompt() -> str:
	# Provide the Document JSON Schema to the LLM and ask for a JSON output
	schema = Document.model_json_schema()
	schema_json = json.dumps(schema, ensure_ascii=False)

	prompt = (
		"You are a data generator. Produce a single JSON object (and ONLY the JSON)\n"
		"with four top-level fields: `date_of_document`, `summary_of_case_till_now`, `document`, and `text`.\n"
		"- `date_of_document` should be a string representing the date the document was written, relative to which the dates in the text should be.\n"
		"- `summary_of_case_till_now` should be a string summarizing the case up to the current document, including a few tumor events (this is a complex case).\n"
		"- `document` must conform to the provided JSON Schema for the Pydantic `Document` model.\n"
		"- `text` must be a short natural-language document (a clinical note) that exactly matches and supports the structured fields in `document`.\n"
		"Use dates in DD/MM/YYYY or YYYY-MM-DD format. For fields taking values, assume the logical unit (mm for tumor size, cm for patient height, kg for patient weight, etc...), do not write it down only write a number. Use the enumerated literal values where applicable.\n"
		"Return valid JSON only — no surrounding markdown, commentary or extra fields at the top level. The `document` field must contain structured fields as in the schema. In `contextsentence` write a realistic sentence which contains that information, and which you will include in the final `text` field, do not use null. In these context sentences, only include text that is natural in a clinical note, avoiding the codes themselves; dates will often be expressed in a relative format such as 'yesterday' or 'last week', relative to the `date_of_document`. Start the document 'text' with a date at which the document was written; format it as a letter from the oncological hospital to the GP of the patient. Include the contextsentence exactly as they were defined above, but feel free to add new sentences.\n\n"
		"Here is the JSON Schema for `Document` (Pydantic):\n"
		f"{schema_json}\n"
	)
	return prompt


def extract_json(text: str) -> Dict[str, Any]:
	# Try to locate the first JSON object in the response
	start = text.find("{")
	end = text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("No JSON object found in model response")
	json_str = text[start : end + 1]
	return json.loads(json_str)


def generate_one_example(client : openai.OpenAI, model: str = "gpt-4o-mini") -> DocumentAndText:
	prompt = build_prompt()

	# Use chat completions with a JSON Schema response_format (supported by vLLM)
	try:
		# Build a combined JSON Schema that keeps Document's $defs at the root
		doc_schema = Document.model_json_schema()
		doc_title = doc_schema.get("title", "Document")
		# Preserve nested defs and also register the top-level Document schema under $defs
		nested_defs = dict(doc_schema.get("$defs", {}))
		# Make a shallow copy of the top-level doc schema without its internal $defs to avoid duplication
		doc_copy = dict(doc_schema)
		doc_copy.pop("$defs", None)
		# Register the top-level Document schema under its title in $defs
		nested_defs[doc_title] = doc_copy

		combined_schema = {
			"$defs": nested_defs,
			"type": "object",
			"properties": {
				"date_of_document": {"type": "string"},
				# Reference the Document definition from the preserved $defs
				"document": {"$ref": f"#/$defs/{doc_title}"},
				"text": {"type": "string"},
			},
			"required": ["date_of_document", "document", "text"],
		}

		resp = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": "You are a data generator that outputs JSON matching the provided schema."},
				{"role": "user", "content": prompt},
			],
			temperature=0.2,
			max_tokens=8192,
			response_format={
				"type": "json_schema",
				"json_schema": {"name": "DocumentAndText", "schema": combined_schema},
			},
		)

		# Try to get structured parse from the response if provided by the server
		data = None
		try:
			# Some servers place parsed result under choices[0].message['response_format']
			choice = resp.choices[0]
			msg = getattr(choice, "message", {}) or {}
			rf = msg.get("response_format") if isinstance(msg, dict) else None
			if rf and isinstance(rf, dict) and rf.get("type") == "json_schema" and "parsed" in rf:
				data = rf.get("parsed")
		except Exception:
			data = None

		# Fallback: extract JSON text from the model content
		if data is None:
			try:
				content = resp.choices[0].message.content
				data = extract_json(content)
			except Exception:
				data = None

		if data is None:
			raise RuntimeError("Could not parse structured response from model")

	except Exception as e:
		print(f"Structured generation failed, falling back to plain chat: {e}", file=sys.stderr)
		# Final fallback: plain chat completion and JSON extraction
		resp = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": "You are a helpful assistant that outputs JSON only."},
				{"role": "user", "content": prompt},
			],
			temperature=0.2,
			max_tokens=8192,
		)
		content = resp.choices[0].message.content
		data = extract_json(content)

	# Validate using Pydantic
	validated = DocumentAndText.model_validate(data)
	return validated


def save_example(obj: DocumentAndText, out_dir: str = "data/generated") -> Path:
	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	# Build a JSON-serializable dict using the model helper to convert dates
	data = {"date_of_document": obj.date_of_document, "document": obj.document.to_json_serializable(), "text": obj.text}

	# Use canonical JSON of the model dump to compute md5
	canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
	md5 = hashlib.md5(canonical.encode("utf-8")).hexdigest()
	file_path = out_path / f"{md5}.json"

	with open(file_path, "w", encoding="utf-8") as fh:
		json.dump(data, fh, indent=2, ensure_ascii=False)

	return file_path


def main():
	parser = argparse.ArgumentParser(description="Generate one Document+text example and save as JSON")
	parser.add_argument("--api-key", help="OpenAI API key (overrides OPENAI_API_KEY)")
	parser.add_argument("--base-url", help="OpenAI API base URL (for vLLM/proxy) e.g. http://localhost:8000/v1")
	parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="Model name to call")
	parser.add_argument("--out-dir", default="data/generated", help="Output directory")
	parser.add_argument("-n", "--number", type=int, default=4, help="Number of documents to generate (default: 4)")
	args = parser.parse_args()

	try:
		client = get_openai_client(api_key=args.api_key, base_url=args.base_url)
	except Exception as e:
		print(f"Error configuring OpenAI client: {e}", file=sys.stderr)
		raise

	# Single generation (no parallelism)
	if args.number <= 1:
		try:
			example = generate_one_example(client=client, model=args.model)
		except Exception as e:
			print(f"Error generating example: {e}", file=sys.stderr)
			raise

		saved = save_example(example, out_dir=args.out_dir)
		print(f"Saved example to: {saved}")
		return

	def worker(idx: int):
		try:
			ex = generate_one_example(client=client, model=args.model)
			path = save_example(ex, out_dir=args.out_dir)
			return path
		except Exception as e:
			return e

	total = args.number
	with ThreadPoolExecutor(max_workers=64) as executor:
		futures = [executor.submit(worker, i) for i in range(total)]

		for fut in tqdm(as_completed(futures), total=total):
			res = fut.result()
			if isinstance(res, Exception):
				print(f"Error generating example: {res}", file=sys.stderr)
			else:
				print(f"Saved example to: {res}")


if __name__ == "__main__":
	main()

