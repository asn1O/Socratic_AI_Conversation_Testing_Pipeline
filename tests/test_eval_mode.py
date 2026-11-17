"""Tests for deterministic Evaluation Mode with sequential modulo parameter selection."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from src.model import Model
from src.pipeline import run_single_conversation
from src.prompt_generator import (
    DEFAULT_EVAL_COMBOS,
    get_eval_combo,
    load_eval_combos,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_eval_combos_defaults():
    combos = load_eval_combos()
    assert isinstance(combos, list)
    assert len(combos) == 10
    assert combos[0]["combo_id"] == "EVAL_01"
    assert combos[-1]["combo_id"] == "EVAL_10"


def test_load_eval_combos_from_json():
    json_path = REPO_ROOT / "eval_combos.json"
    assert json_path.is_file()
    combos = load_eval_combos(str(json_path))
    assert len(combos) == 10
    assert combos[0]["scenario_id"] == "PHY_001"
    assert combos[0]["persona_id"] == "STU_002"
    assert combos[1]["scenario_id"] == "MAT_001"
    assert combos[1]["persona_id"] == "STU_004"


def test_load_eval_combos_fallback_on_invalid(tmp_path):
    missing_file = tmp_path / "non_existent.json"
    combos = load_eval_combos(str(missing_file))
    assert len(combos) == len(DEFAULT_EVAL_COMBOS)
    assert combos[0]["combo_id"] == "EVAL_01"


def test_get_eval_combo_sequential_modulo_1indexed():
    """Verify that conversation index i (1-indexed) selects (i-1) % len(combos)th combo."""
    combos = load_eval_combos()
    assert len(combos) == 10

    # 1st convo -> 1st combo (EVAL_01)
    c1 = get_eval_combo(1, one_indexed=True, combos=combos)
    assert c1["combo_id"] == "EVAL_01"
    assert c1["scenario_id"] == "PHY_001"
    assert c1["persona_id"] == "STU_002"

    # 2nd convo -> 2nd combo (EVAL_02)
    c2 = get_eval_combo(2, one_indexed=True, combos=combos)
    assert c2["combo_id"] == "EVAL_02"
    assert c2["scenario_id"] == "MAT_001"

    # 10th convo -> 10th combo (EVAL_10)
    c10 = get_eval_combo(10, one_indexed=True, combos=combos)
    assert c10["combo_id"] == "EVAL_10"
    assert c10["scenario_id"] == "LOG_002"

    # 11th convo -> 1st combo (EVAL_01) (11 mod 10 = 1)
    c11 = get_eval_combo(11, one_indexed=True, combos=combos)
    assert c11["combo_id"] == "EVAL_01"
    assert c11["scenario_id"] == "PHY_001"

    # 12th convo -> 2nd combo (EVAL_02) (12 mod 10 = 2)
    c12 = get_eval_combo(12, one_indexed=True, combos=combos)
    assert c12["combo_id"] == "EVAL_02"
    assert c12["scenario_id"] == "MAT_001"

    # 25th convo -> 5th combo (EVAL_05) (25 mod 10 = 5)
    c25 = get_eval_combo(25, one_indexed=True, combos=combos)
    assert c25["combo_id"] == "EVAL_05"
    assert c25["scenario_id"] == "LOG_001"


def test_get_eval_combo_0indexed():
    combos = load_eval_combos()
    assert get_eval_combo(0, one_indexed=False, combos=combos)["combo_id"] == "EVAL_01"
    assert get_eval_combo(9, one_indexed=False, combos=combos)["combo_id"] == "EVAL_10"
    assert get_eval_combo(10, one_indexed=False, combos=combos)["combo_id"] == "EVAL_01"
    assert get_eval_combo(11, one_indexed=False, combos=combos)["combo_id"] == "EVAL_02"


def test_run_single_conversation_eval_mode(tmp_path):
    tutor = Model(
        provider="Google",
        api_key="test",
        model_name="gemini-3.5-flash",
        url="http://localhost",
    )
    student = Model(
        provider="Google",
        api_key="test",
        model_name="gemini-2.5-flash-lite",
        url="http://localhost",
    )

    data_dir = tmp_path / "data"

    # Run for convo index 1 -> Should select EVAL_01 (PHY_001, STU_002)
    success1, save_path1, record1 = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=5,
        data_dir=str(data_dir),
        mock=True,
        eval_mode=True,
        eval_combo_index=1,
    )
    assert success1 is True
    assert record1["config"]["eval_mode"] is True
    assert record1["config"]["eval_combo_id"] == "EVAL_01"
    assert record1["scenario_id"] == "PHY_001"
    assert record1["config"]["parameters"]["persona_id"] == "STU_002"

    # Run for convo index 11 -> Should wrap around to EVAL_01 (PHY_001, STU_002)
    success11, save_path11, record11 = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=5,
        data_dir=str(data_dir),
        mock=True,
        eval_mode=True,
        eval_combo_index=11,
    )
    assert success11 is True
    assert record11["config"]["eval_combo_id"] == "EVAL_01"
    assert record11["scenario_id"] == "PHY_001"
    assert record11["config"]["parameters"]["persona_id"] == "STU_002"

    # Run for convo index 12 -> Should wrap around to EVAL_02 (MAT_001, STU_004)
    success12, save_path12, record12 = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=5,
        data_dir=str(data_dir),
        mock=True,
        eval_mode=True,
        eval_combo_index=12,
    )
    assert success12 is True
    assert record12["config"]["eval_combo_id"] == "EVAL_02"
    assert record12["scenario_id"] == "MAT_001"
    assert record12["config"]["parameters"]["persona_id"] == "STU_004"


def test_cli_eval_mode_modulo_sequence(tmp_path):
    """Test CLI runner with --eval-mode across 12 iterations in mock mode."""
    data_dir = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_pipeline.py"),
            "--mock",
            "--eval-mode",
            "--iterations", "12",
            "--data-dir", str(data_dir),
            "--log-file", "none",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert "EVAL MODE (10 sequential benchmark combos)" in stdout
    assert "PIPELINE RUN SUMMARY" in stdout
    assert "Mode                  : EVAL MODE" in stdout
    assert "Total Sessions        : 12" in stdout
    assert "Successfully Stored   : 12" in stdout

    # Verify session 1 and 11 used EVAL_01
    assert "Combo 1/10 (EVAL_01)" in stdout
    # Verify session 12 used EVAL_02
    assert "Combo 2/10 (EVAL_02)" in stdout
    # Verify session 10 used EVAL_10
    assert "Combo 10/10 (EVAL_10)" in stdout
