"""Tests for turn-level and conversation dataset validators."""

import json
from pathlib import Path
import pytest

from src.validator import (
    check_config_leakage,
    is_valid_conversation_dataset,
    validate_student_turn,
    validate_tutor_turn,
)


def test_validate_tutor_turn_valid():
    sample = json.dumps({
        "reasoning": "The student has an initial misconception. Guiding with an open question.",
        "end": 0,
        "response": "How did you arrive at that calculation?",
    })
    is_valid, parsed, err = validate_tutor_turn(sample)
    assert is_valid is True
    assert err == ""
    assert parsed["end"] == 0
    assert parsed["reasoning"].startswith("The student has")
    assert parsed["response"].startswith("How did you")


def test_validate_tutor_turn_markdown_fences():
    sample = "```json\n" + json.dumps({
        "reasoning": "Student articulated rule correctly. Wrap up.",
        "end": 1,
        "response": "Well done! You got it.",
    }) + "\n```"
    is_valid, parsed, err = validate_tutor_turn(sample)
    assert is_valid is True
    assert parsed["end"] == 1


def test_validate_tutor_turn_invalid_end():
    sample = json.dumps({
        "reasoning": "Some reasoning.",
        "end": 5,  # Must be 0 or 1
        "response": "Listen to me.",
    })
    is_valid, parsed, err = validate_tutor_turn(sample)
    assert is_valid is False
    assert "invalid 'end' field" in err


def test_validate_tutor_turn_missing_end():
    sample = json.dumps({
        "reasoning": "Some reasoning.",
        "response": "Listen to me.",
    })
    is_valid, parsed, err = validate_tutor_turn(sample)
    assert is_valid is False
    assert "missing required field 'end'" in err


def test_validate_tutor_turn_missing_reasoning():
    sample = json.dumps({
        "end": 0,
        "response": "Try dividing by 2.",
    })
    is_valid, parsed, err = validate_tutor_turn(sample)
    assert is_valid is False
    assert "missing or empty 'reasoning'" in err


def test_validate_tutor_turn_config_leakage():
    # Direct parameter leakage
    sample_leak = json.dumps({
        "reasoning": "Student needs help as per low socratic_level and high verbosity.",
        "end": 0,
        "response": "Here is a hint.",
    })
    is_valid, parsed, err = validate_tutor_turn(sample_leak)
    assert is_valid is False
    assert "config leakage detected" in err


def test_validate_tutor_turn_persona_leakage():
    sample_leak = json.dumps({
        "reasoning": "Since this is a beginner student, I will be gentle.",
        "end": 0,
        "response": "Let's take it step by step.",
    })
    is_valid, parsed, err = validate_tutor_turn(sample_leak)
    assert is_valid is False
    assert "config leakage detected" in err


def test_check_config_leakage_allowed_difficulty():
    # General English difficulty usage should be permitted
    allowed_reasoning = "The student has difficulty understanding the concept of ratios."
    flags = check_config_leakage(allowed_reasoning)
    assert len(flags) == 0

    # Config difficulty leak should be flagged
    leaked_reasoning = "I will maintain the 'hard' difficulty setting here."
    flags_leaked = check_config_leakage(leaked_reasoning)
    assert len(flags_leaked) > 0


def test_validate_student_turn():
    valid, text, err = validate_student_turn("I think the answer is 5.")
    assert valid is True
    assert text == "I think the answer is 5."
    assert err == ""

    valid_empty, _, err_empty = validate_student_turn("   ")
    assert valid_empty is False
    assert "empty" in err_empty


def test_is_valid_conversation_dataset():
    good_convo = {
        "problem": "Solve for x in 2x = 10",
        "conversation": [
            {"turn": 1, "speaker": "student", "text": "Is it 20?"},
            {"turn": 2, "speaker": "tutor", "reasoning": "Student multiplied instead of dividing.", "end": 0, "response": "Think about the inverse operation."},
            {"turn": 3, "speaker": "student", "text": "Oh, divide by 2, so x = 5!"},
            {"turn": 4, "speaker": "tutor", "reasoning": "Student got it. Ask to generalize.", "end": 0, "response": "In general, what do you do when a variable is multiplied by a coefficient?"},
            {"turn": 5, "speaker": "student", "text": "You divide both sides by the coefficient."},
            {"turn": 6, "speaker": "tutor", "reasoning": "Student explained rule correctly.", "end": 1, "response": "Exactly right! Great work."},
        ]
    }
    is_valid, detail, category = is_valid_conversation_dataset(good_convo)
    assert is_valid is True
    assert detail == ""

    # Test bad ending (last tutor turn has end: 0)
    bad_convo = {
        "problem": "Solve for x",
        "conversation": [
            {"turn": 1, "speaker": "student", "text": "x is 5"},
            {"turn": 2, "speaker": "tutor", "reasoning": "Good.", "end": 0, "response": "Nice"},
            {"turn": 3, "speaker": "student", "text": "thanks"},
            {"turn": 4, "speaker": "tutor", "reasoning": "Bye.", "end": 0, "response": "Bye"},
        ]
    }
    is_valid_bad, detail_bad, _ = is_valid_conversation_dataset(bad_convo)
    assert is_valid_bad is False
    assert "end" in detail_bad or "ended" in detail_bad
