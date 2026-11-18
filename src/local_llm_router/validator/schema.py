"""JSON extraction and schema validation utilities."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import jsonschema


def extract_json_candidate(text: str) -> Optional[str]:
    """Extract JSON string from text, stripping markdown codeblocks if present."""
    text_clean = text.strip()

    # Match ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text_clean, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # If starts with { or [, locate the outermost matching bracket
    first_brace = text_clean.find("{")
    first_bracket = text_clean.find("[")

    start_idx = -1
    if first_brace != -1 and first_bracket != -1:
        start_idx = min(first_brace, first_bracket)
    elif first_brace != -1:
        start_idx = first_brace
    elif first_bracket != -1:
        start_idx = first_bracket

    if start_idx != -1:
        last_brace = text_clean.rfind("}")
        last_bracket = text_clean.rfind("]")
        end_idx = max(last_brace, last_bracket)
        if end_idx > start_idx:
            return text_clean[start_idx : end_idx + 1]

    return text_clean


def parse_and_validate_json(
    raw_content: str,
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Any], List[str]]:
    """Parse JSON and validate against JSON schema.

    Returns:
        (is_valid, parsed_data, list_of_error_strings)
    """
    candidate = extract_json_candidate(raw_content)
    if not candidate:
        return False, None, ["Empty output or no JSON found in response"]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        return False, None, [f"JSON syntax error: {e.msg} at line {e.lineno}, column {e.colno}"]

    if not schema:
        return True, parsed, []

    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(parsed), key=lambda e: e.path):
        path_str = "/" + "/".join(str(p) for p in err.path) if err.path else "root"
        errors.append(f"At {path_str}: {err.message}")

    if errors:
        return False, parsed, errors

    return True, parsed, []
