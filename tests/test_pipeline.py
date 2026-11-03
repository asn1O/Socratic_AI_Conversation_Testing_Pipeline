"""End-to-end tests for pipeline conversation loop and model loading."""

import json
from pathlib import Path
import pytest

from src.model import Model
from src.pipeline import (
    get_model_by_index_or_name,
    load_models_from_file,
    run_single_conversation,
)


def test_load_models_and_selection(tmp_path):
    models_file = tmp_path / "models.json"
    models_data = [
        {"provider": "Google", "model_name": "gemini-3.5-flash", "api_key": "TEST_KEY", "url": "http://localhost"},
        {"provider": "NVIDIA", "model_name": "nemotron-3", "api_key": "TEST_KEY", "url": "http://localhost"},
    ]
    models_file.write_text(json.dumps(models_data), encoding="utf-8")

    models = load_models_from_file(str(models_file))
    assert len(models) == 2

    # Select by index
    m0 = get_model_by_index_or_name(models, index=0)
    assert m0.model_name == "gemini-3.5-flash"
    m1 = get_model_by_index_or_name(models, index=1)
    assert m1.model_name == "nemotron-3"

    # Select by name
    m_name = get_model_by_index_or_name(models, name="nemotron")
    assert m_name.model_name == "nemotron-3"


def test_run_single_conversation_mock(tmp_path):
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

    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=5,
        data_dir=str(data_dir),
        generator_id="test_gen",
        mock=True,
    )

    assert success is True
    assert save_path is not None
    assert save_path.exists()

    # Check saved data
    with save_path.open("r", encoding="utf-8") as f:
        saved_items = json.load(f)

    assert len(saved_items) == 1
    item = saved_items[0]
    convo = item["conversation"]

    # 3 mock rounds -> 6 turns total
    assert len(convo) == 6
    assert convo[0]["speaker"] == "student"
    assert convo[1]["speaker"] == "tutor"
    assert convo[-1]["speaker"] == "tutor"
    assert convo[-1]["state"] == "Closing"
    assert item["config"]["closed"] is True

    # Verify persona and question bank details are recorded in parameters
    params = item["config"]["parameters"]
    assert params["persona_id"].startswith("STU_")
    assert isinstance(params["persona_name"], str) and len(params["persona_name"]) > 0
    assert "scenario_id" in params
    assert "difficulty" in params
    assert "learning_objective" in params
    assert "target_misconception" in params
    assert item["scenario_id"] == params["scenario_id"]


def test_run_single_conversation_custom_scenario_and_persona(tmp_path):
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

    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=5,
        scenario_id="PHY_001",
        persona_id="STU_002",
        data_dir=str(data_dir),
        generator_id="test_gen",
        mock=True,
    )

    assert success is True
    assert record["scenario_id"] == "PHY_001"
    assert record["domain"] == "Physics"
    params = record["config"]["parameters"]
    assert params["persona_id"] == "STU_002"
    assert params["persona_name"] == "Confident but Incorrect"
    assert params["scenario_id"] == "PHY_001"
    assert params["difficulty"] == "medium"
    assert "Newton's Third Law" in params["learning_objective"]


def test_run_single_conversation_random_keyword(tmp_path):
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

    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=3,
        scenario_id="random",
        persona_id="random",
        data_dir=str(data_dir),
        generator_id="test_gen",
        mock=True,
    )

    assert success is True
    params = record["config"]["parameters"]
    assert params["persona_id"].startswith("STU_")
    assert record["scenario_id"] == params["scenario_id"]
    assert len(params["scenario_id"]) > 0


def test_set_file_splitting(tmp_path):
    tutor = Model(
        provider="Google",
        api_key="test",
        model_name="gemini-3.5-flash",
        url="http://localhost",
    )
    data_dir = tmp_path / "data"

    # Append 16 items to verify set[0].json holds 15 and set[1].json holds 1
    last_path = None
    for i in range(16):
        sample_convo = {
            "config": {"id": i},
            "problem": f"Problem {i}",
            "conversation": [{"turn": 1, "speaker": "student", "text": "hi"}],
        }
        last_path = tutor.write_response(sample_convo, data_dir=str(data_dir), generator_id="asn")

    stem = tutor.response_file_stem
    set0_path = data_dir / stem / "asn-set[0].json"
    set1_path = data_dir / stem / "asn-set[1].json"

    assert set0_path.exists()
    assert set1_path.exists()

    with set0_path.open("r", encoding="utf-8") as f:
        data0 = json.load(f)
    with set1_path.open("r", encoding="utf-8") as f:
        data1 = json.load(f)

    assert len(data0) == 15
    assert len(data1) == 1


def test_run_single_conversation_student_retry_failure(tmp_path, monkeypatch):
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

    calls = 0
    def mock_student_call(prompt, **kwargs):
        nonlocal calls
        calls += 1
        return False, None, 0, "Rate limit error"

    monkeypatch.setattr(student, "call_model", mock_student_call)

    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=5,
        max_student_retries=5,
        data_dir=str(data_dir),
        generator_id="test_gen",
        mock=False,
    )

    assert success is False
    assert save_path is None
    assert calls == 5
    assert "student failed after 5 retries on turn 1" in record["error"]

    # Verify rejected response was written to data folder
    stem = tutor.response_file_stem
    rejected_file = data_dir / "rejected_responses" / f"test_gen-{stem}.txt"
    assert rejected_file.exists()
    content = rejected_file.read_text(encoding="utf-8")
    assert "Student failed after 5 retries on turn 1" in content


def test_run_single_conversation_student_retry_success(tmp_path, monkeypatch):
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

    student_calls = 0
    def mock_student_call(prompt, **kwargs):
        nonlocal student_calls
        student_calls += 1
        if student_calls < 3:
            return False, None, 0, "Temporary network timeout"
        return True, "I think the answer is 10 cents.", 15, None

    tutor_calls = 0
    def mock_tutor_call(prompt, **kwargs):
        nonlocal tutor_calls
        tutor_calls += 1
        return True, json.dumps({
            "reasoning": "The student gave 10 cents.",
            "state": "Closing",
            "response": "Good job!",
        }), 20, None

    monkeypatch.setattr(student, "call_model", mock_student_call)
    monkeypatch.setattr(tutor, "call_model", mock_tutor_call)

    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=3,
        max_student_retries=5,
        data_dir=str(data_dir),
        generator_id="test_gen",
        mock=False,
    )

    assert success is True
    assert save_path is not None
    assert student_calls == 3  # Failed 2 times, succeeded on 3rd
    assert len(record["conversation"]) == 2


def test_run_single_conversation_log_file(tmp_path):
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
    log_file = tmp_path / "custom_logs" / "test_run.log"

    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=3,
        data_dir=str(data_dir),
        generator_id="test_gen",
        mock=True,
        log_file=str(log_file),
    )

    assert success is True
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Starting conversation test" in content
    assert "COMPLETE CONVERSATION TRANSCRIPT" in content
    assert "Turn 1 [STUDENT" in content
    assert "Turn 2 [TUTOR" in content


def test_run_single_conversation_tutor_prompt_default_and_custom(tmp_path):
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

    # 1. Default tutor prompt -> socratic_tutor_prompt_small.md
    success_def, _, record_def = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=2,
        data_dir=str(data_dir),
        generator_id="test_default",
        mock=True,
    )
    assert success_def is True
    assert record_def["config"]["tutor_prompt"] == "socratic_tutor_prompt_small.md"

    # 2. Custom tutor prompt -> socratic_tutor_prompt.md
    success_custom, _, record_custom = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=2,
        data_dir=str(data_dir),
        generator_id="test_custom",
        mock=True,
        tutor_prompt_path="socratic_tutor_prompt.md",
    )
    assert success_custom is True
    assert record_custom["config"]["tutor_prompt"] == "socratic_tutor_prompt.md"


def test_run_pipeline_cli_tutor_prompt_arg(monkeypatch):
    import subprocess
    import sys
    from scripts.run_pipeline import PROJECT_ROOT

    # Run run_pipeline.py with --help to verify argument is present
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--tutor-prompt" in result.stdout
    assert "socratic_tutor_prompt_small.md" in result.stdout

    # Run with default tutor prompt in mock mode
    result_mock = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--mock",
            "--iterations", "1",
            "--max-iterations", "2",
            "--log-file", "none",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result_mock.returncode == 0
    assert "Tutor Prompt  : socratic_tutor_prompt_small.md" in result_mock.stdout

    # Run with custom tutor prompt in mock mode
    result_custom = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--mock",
            "--tutor-prompt", "socratic_tutor_prompt.md",
            "--iterations", "1",
            "--max-iterations", "2",
            "--log-file", "none",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result_custom.returncode == 0
    assert "Tutor Prompt  : socratic_tutor_prompt.md" in result_custom.stdout

    # Run with invalid tutor prompt file -> error
    result_invalid = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--tutor-prompt", "non_existent_file.md",
            "--mock",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result_invalid.returncode != 0
    assert "Tutor prompt file not found: non_existent_file.md" in result_invalid.stderr



