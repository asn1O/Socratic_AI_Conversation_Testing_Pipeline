"""Validation utilities for turn-level responses and complete dataset conversations."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

VALID_TUTOR_STATES = {"diagnosis", "support", "synthesis", "closing"}
VALID_SPEAKERS = {"student", "tutor"}

_THOUGHT_BLOCK_PATTERN = re.compile(
    r"(?is)<(think|thought|scratchpad|reflection)\b[^>]*>.*?</\1>"
)

# Config leakage regex patterns derived from flagged_config_leakage.json
_DIRECT_PARAM_PATTERNS = [
    re.compile(r"\b(socratic_level|hint_frequency|mistake_frequency|student_level)\b", re.IGNORECASE),
    re.compile(r"\b(socratic\s+level|hint\s+frequency|mistake\s+frequency|student\s+level)\b", re.IGNORECASE),
    re.compile(r"\b(beginner|intermediate|advanced)\s+(student|persona|level)\b", re.IGNORECASE),
    re.compile(r"\bas\s+per\s+the\s+.*?(level|frequency|config|socratic|verbosity)\b", re.IGNORECASE),
    re.compile(r"\bverbosity\b", re.IGNORECASE),
]

_DIFFICULTY_LEAK_PATTERN = re.compile(
    r"\b(easy|medium|hard)\s+difficulty\b|\bdifficulty\s*:\s*(easy|medium|hard)\b|\bmaintain\s+the\s+['\"]?(easy|medium|hard)['\"]?\s+difficulty\b",
    re.IGNORECASE,
)


def strip_wrappers(raw_text: str) -> str:
    """Remove markdown fences and reasoning scratchpad tags from model output."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = _THOUGHT_BLOCK_PATTERN.sub("", cleaned)
    return cleaned.strip()


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """Parse the first valid JSON object from text using raw_decode."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")

    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text, idx=start)
    if not isinstance(payload, dict):
        raise ValueError("parsed JSON is not an object")
    return payload


def check_config_leakage(text: str, custom_parameters: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Check text (especially tutor reasoning) for prompt configuration leakage.
    Returns a list of matched leakage descriptions.
    """
    if not text:
        return []

    flags = []

    for pattern in _DIRECT_PARAM_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            flags.append(f"direct_parameter_or_persona_leak: {matches}")

    difficulty_matches = _DIFFICULTY_LEAK_PATTERN.findall(text)
    if difficulty_matches:
        flags.append(f"difficulty_config_leak: {difficulty_matches}")

    # Check custom parameters if provided (skip standard metadata fields)
    standard_meta_keys = {
        "difficulty", "scenario_id", "persona_id", "persona_name",
        "learning_objective", "target_misconception", "expected_concepts",
        "problem", "question", "topic", "subtopic", "domain",
    }
    if custom_parameters:
        for param_name, param_val in custom_parameters.items():
            if param_name.lower() in standard_meta_keys:
                continue
            if param_name.lower() in text.lower():
                flags.append(f"custom_param_name_leak: {param_name}")
            if isinstance(param_val, str) and len(param_val) > 4 and param_val.lower() in text.lower():
                # Avoid single small words, check distinctive values
                flags.append(f"custom_param_val_leak: {param_val}")

    return flags


def validate_tutor_turn(
    reply_text: str,
    custom_parameters: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Validate a single tutor turn response.

    Returns:
        (is_valid, parsed_turn_dict, error_message)
    """
    cleaned = strip_wrappers(reply_text)
    if not cleaned:
        return False, None, "empty response"

    try:
        payload = extract_first_json_object(cleaned)
    except Exception as exc:
        return False, None, f"invalid JSON: {exc}"

    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False, None, "missing or empty 'reasoning' field"

    end_val = payload.get("end")
    if end_val is None:
        return False, None, "missing required field 'end' (must be 0 or 1)"

    if isinstance(end_val, bool):
        normalized_end = 1 if end_val else 0
    elif isinstance(end_val, int) and end_val in (0, 1):
        normalized_end = end_val
    elif isinstance(end_val, str) and end_val.strip() in ("0", "1"):
        normalized_end = int(end_val.strip())
    else:
        return False, None, f"invalid 'end' field: must be 0 or 1 (got: {end_val})"

    response = payload.get("response") or payload.get("text")
    if not isinstance(response, str) or not response.strip():
        return False, None, "missing or empty 'response' field"

    # Check for config leakage in reasoning
    leakage_flags = check_config_leakage(reasoning, custom_parameters)
    if leakage_flags:
        return False, None, f"config leakage detected in reasoning: {', '.join(leakage_flags)}"

    valid_payload = {
        "reasoning": reasoning.strip(),
        "end": normalized_end,
        "response": response.strip(),
    }
    return True, valid_payload, ""


def validate_student_turn(
    reply_text: str,
    custom_parameters: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Validate a single student turn response.
    Extensible hook for future student persona / parameter validation.

    Returns:
        (is_valid, cleaned_text, error_message)
    """
    cleaned = (reply_text or "").strip()
    # Strip wrapping quotes or markdown if the model wrapped its output
    cleaned = _THOUGHT_BLOCK_PATTERN.sub("", cleaned).strip()

    if not cleaned:
        return False, "", "empty student response"

    # Extensible hook: check student response for prompt leakage once parameters are added
    if custom_parameters:
        leakage_flags = check_config_leakage(cleaned, custom_parameters)
        if leakage_flags:
            return False, cleaned, f"student config leakage: {', '.join(leakage_flags)}"

    return True, cleaned, ""


def is_valid_conversation_dataset(entry: dict) -> Tuple[bool, str, str]:
    """
    Standalone validator to check conversation records in data folder.

    Validates:
      1. Required fields: 'problem', 'conversation'
      2. Turn numbering: 1..N
      3. Speaker alternation: student -> tutor -> ...
      4. Tutor fields: reasoning, end (0 or 1), response
      5. Last tutor turn must have end: 1
      6. No config leakage in reasoning

    Returns:
        (is_valid, error_detail, error_category)
    """
    if "problem" not in entry:
        return False, "missing 'problem' field", "missing required field"
    if "conversation" not in entry:
        return False, "missing 'conversation' field", "missing required field"

    turns = entry.get("conversation")
    if not isinstance(turns, list) or len(turns) == 0:
        return False, "'conversation' is not a non-empty list", "invalid conversation format"

    tutor_turns: List[Tuple[int, Dict[str, Any]]] = []

    for i, turn in enumerate(turns):
        expected_turn = i + 1
        if turn.get("turn") != expected_turn:
            return False, f"turn {i+1}: expected turn number {expected_turn}, got {turn.get('turn')}", "incorrect turn numbering"

        speaker = str(turn.get("speaker", "")).strip().lower()
        if speaker not in VALID_SPEAKERS:
            return False, f"turn {i+1}: invalid speaker '{turn.get('speaker')}'", "invalid speaker value"

        expected_speaker = "student" if i % 2 == 0 else "tutor"
        if speaker != expected_speaker:
            return False, f"turn {i+1}: expected '{expected_speaker}', got '{speaker}'", "speaker alternation broken"

        if speaker == "tutor":
            tutor_turns.append((i, turn))
            end_val = turn.get("end")
            if end_val not in (0, 1, "0", "1", True, False):
                return False, f"turn {i+1} (tutor): invalid or missing 'end' field '{end_val}'", "invalid tutor end value"

            reasoning = str(turn.get("reasoning", ""))
            if not reasoning.strip():
                return False, f"turn {i+1} (tutor): missing or empty 'reasoning'", "missing required field"

            response = str(turn.get("response", turn.get("text", "")))
            if not response.strip():
                return False, f"turn {i+1} (tutor): missing or empty 'response'", "missing required field"

            leakage = check_config_leakage(reasoning)
            if leakage:
                return False, f"turn {i+1} (tutor): config leakage detected ({leakage})", "config leakage"

    if len(tutor_turns) < 1:
        return False, "no tutor turns found", "too few tutor turns"

    last_tutor_idx, last_tutor = tutor_turns[-1]
    last_end = last_tutor.get("end")
    if last_end not in (1, "1", True):
        return (
            False,
            f"turn {last_tutor_idx+1}: last tutor turn must have 'end': 1, got '{last_end}'",
            "last tutor turn not ended",
        )

    return True, "", ""
