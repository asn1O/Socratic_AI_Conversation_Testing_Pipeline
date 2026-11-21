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
    assert convo[-1]["end"] == 1
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
            "end": 1,
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


def test_run_pipeline_cli_default_log_and_summary(tmp_path):
    import subprocess
    import sys
    from scripts.run_pipeline import PROJECT_ROOT

    # Run in mock mode without specifying --log-file; should default to logs/{tutor_stem}.log
    data_dir = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--mock",
            "--iterations", "2",
            "--max-iterations", "2",
            "--data-dir", str(data_dir),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Output should include final summary with counts
    assert "PIPELINE RUN SUMMARY" in result.stdout
    assert "Total Sessions        : 2" in result.stdout
    assert "Successfully Stored   : 2" in result.stdout
    assert "Failed Sessions       : 0" in result.stdout
    assert "Success Rate          : 100.0%" in result.stdout

    # Verify log file was created using provider and model name (tutor 0 is Google gemini-3.5-flash)
    expected_log = PROJECT_ROOT / "logs" / "NVIDIA_nvidia_nemotron-3.5-super-vl-120b-a12b.log"
    # Actually, check what tutor 0 is in tutors.json
    # tutors.json has NVIDIA_nvidia_nemotron-3.5-super-vl-120b-a12b as index 0!
    assert expected_log.exists()
    content = expected_log.read_text(encoding="utf-8")
    assert "PIPELINE RUN SUMMARY" in content
    assert "Total Sessions        : 2" in content
    assert "Successfully Stored   : 2" in content


def test_run_pipeline_cli_failure_summary(tmp_path):
    import subprocess
    import sys
    from scripts.run_pipeline import PROJECT_ROOT

    # Run in mock mode with require-closing and max-iterations 1 (will fail because mock requires 3 rounds for closing)
    data_dir = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--mock",
            "--iterations", "2",
            "--max-iterations", "1",
            "--require-closing",
            "--data-dir", str(data_dir),
            "--log-file", "none",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "PIPELINE RUN SUMMARY" in result.stdout
    assert "Total Sessions        : 2" in result.stdout
    assert "Successfully Stored   : 0" in result.stdout
    assert "Failed Sessions       : 2" in result.stdout
    assert "Success Rate          : 0.0%" in result.stdout
    assert "Failure Breakdown     :" in result.stdout
    assert "did not reach end: 1" in result.stdout


def test_run_single_conversation_finetuned_socratic_mock(tmp_path):
    from src.validator import is_valid_conversation_dataset

    tutor = Model(
        provider="Kaggle",
        api_key="test",
        model_name="qwen",
        url="http://localhost",
        tutor_type="finetuned_socratic",
        prompt_format="sft_xml",
    )
    assert tutor.is_finetuned_socratic is True

    student = Model(
        provider="NVIDIA",
        api_key="test",
        model_name="nemotron-3",
        url="http://localhost",
    )

    data_dir = tmp_path / "data"

    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=5,
        data_dir=str(data_dir),
        generator_id="ft_test",
        mock=True,
    )

    assert success is True
    assert save_path is not None
    assert save_path.exists()

    # Check configuration recording
    assert record["config"]["tutor_type"] == "finetuned_socratic"
    assert record["config"]["tutor_prompt"] == "finetuned_socratic_xml"
    assert record["config"]["closed"] is True

    # 3 mock rounds -> 6 turns total
    convo = record["conversation"]
    assert len(convo) == 6

    # Verify tutor turns contain state and end mappings
    tutor_turns = [t for t in convo if t["speaker"] == "tutor"]
    assert len(tutor_turns) == 3

    assert tutor_turns[0]["turn"] == 2
    assert tutor_turns[0]["state"] == "Diagnosis"
    assert tutor_turns[0]["end"] == 0

    assert tutor_turns[1]["turn"] == 4
    assert tutor_turns[1]["state"] == "Support"
    assert tutor_turns[1]["end"] == 0

    assert tutor_turns[2]["turn"] == 6
    assert tutor_turns[2]["state"] == "Closing"
    assert tutor_turns[2]["end"] == 1

    # Verify saved conversation is 100% structurally valid under is_valid_conversation_dataset
    is_valid, err_detail, _ = is_valid_conversation_dataset(record)
    assert is_valid is True, f"Dataset validation failed: {err_detail}"

    # Verify JSON file on disk
    with save_path.open("r", encoding="utf-8") as f:
        saved_records = json.load(f)
    assert len(saved_records) == 1
    assert saved_records[0]["conversation"][-1]["state"] == "Closing"
    assert saved_records[0]["conversation"][-1]["end"] == 1


def test_run_pipeline_cli_finetuned_tutor(tmp_path):
    import subprocess
    import sys
    from scripts.run_pipeline import PROJECT_ROOT

    data_dir = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
            "--mock",
            "--tutor-model", "ft",
            "--iterations", "1",
            "--max-iterations", "5",
            "--data-dir", str(data_dir),
            "--log-file", "none",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Tutor Model   : ft - qwen-ft" in result.stdout
    assert "Successfully Stored   : 1" in result.stdout
    assert "Failed Sessions       : 0" in result.stdout


def test_tutors_json_qwen_models_structure():
    from src.pipeline import load_models_from_file
    from scripts.run_pipeline import PROJECT_ROOT

    tutors = load_models_from_file(str(PROJECT_ROOT / "models" / "tutors.json"))
    assert len(tutors) == 8

    # First 4 are NVIDIA and OpenRouter
    assert tutors[0].provider == "NVIDIA"
    assert tutors[1].provider == "NVIDIA"
    assert tutors[2].provider == "OpenRouter"
    assert tutors[3].provider == "OpenRouter"

    # Last 4 are the newly configured Qwen models
    m_base = tutors[4]
    assert m_base.provider == "base"
    assert m_base.model_name == "qwen-base"
    assert m_base.tutor_prompt == "base_tutor_prompt.md"
    assert m_base.history_window == 8

    m_socratic = tutors[5]
    assert m_socratic.provider == "socratic"
    assert m_socratic.model_name == "qwen-base"
    assert m_socratic.tutor_prompt == "socratic_tutor_prompt_small.md"
    assert m_socratic.history_window == 8

    m_ft = tutors[6]
    assert m_ft.provider == "ft"
    assert m_ft.model_name == "qwen-ft"
    assert m_ft.is_finetuned_socratic is True
    assert m_ft.history_window == 8

    m_socratic_ft = tutors[7]
    assert m_socratic_ft.provider == "socratic_ft"
    assert m_socratic_ft.model_name == "qwen-ft"
    assert m_socratic_ft.is_finetuned_socratic is True
    assert m_socratic_ft.history_window == 8
    assert m_socratic_ft.system_prompt == "pedagogical"


def test_run_single_conversation_history_window_in_config(tmp_path):
    from src.pipeline import run_single_conversation
    from src.model import Model

    tutor = Model(
        provider="base",
        api_key="test",
        model_name="qwen-base",
        url="http://localhost",
        history_window=8,
    )
    student = Model(
        provider="NVIDIA",
        api_key="test",
        model_name="nvidia/nemotron-3-ultra-550b-a55b",
        url="http://localhost",
    )

    data_dir = tmp_path / "data"
    success, save_path, record = run_single_conversation(
        tutor_model=tutor,
        student_model=student,
        max_iterations=3,
        data_dir=str(data_dir),
        mock=True,
    )

    assert success is True
    assert save_path.is_file()
    assert record["config"]["history_window"] == 8
    assert record["config"]["tutor_provider"] == "base"


def test_cli_runs_all_4_qwen_models_mock(tmp_path):
    import subprocess
    import sys
    from scripts.run_pipeline import PROJECT_ROOT

    data_dir = tmp_path / "data"
    # Test indices 4, 5, 6, 7
    expected_providers = [
        (4, "base", "qwen-base"),
        (5, "socratic", "qwen-base"),
        (6, "ft", "qwen-ft"),
        (7, "socratic_ft", "qwen-ft"),
    ]

    for idx, provider, model_name in expected_providers:
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
                "--mock",
                "--tutor-index", str(idx),
                "--student-index", "2",
                "--iterations", "1",
                "--max-iterations", "3",
                "--data-dir", str(data_dir),
                "--log-file", "none",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed for tutor index {idx}: {result.stderr}"
        assert f"Tutor Model   : {provider} - {model_name}" in result.stdout
        assert "Student Model : NVIDIA - nvidia/nemotron-3-ultra-550b-a55b" in result.stdout
        assert "History Window: 8" in result.stdout
        assert "Successfully Stored   : 1" in result.stdout


def test_pipeline_differentiates_ft_and_socratic_ft_system_prompts(tmp_path):
    from src.pipeline import load_models_from_file, run_single_conversation
    from scripts.run_pipeline import PROJECT_ROOT

    tutors = load_models_from_file(str(PROJECT_ROOT / "models" / "tutors.json"))
    students = load_models_from_file(str(PROJECT_ROOT / "models" / "students.json"))

    tutor_ft = tutors[6]          # provider: "ft"
    tutor_socratic_ft = tutors[7] # provider: "socratic_ft", system_prompt: "pedagogical"
    student = students[2]

    data_dir = tmp_path / "data"

    # Test ft -> minimal system prompt label
    success, _, record_ft = run_single_conversation(
        tutor_model=tutor_ft,
        student_model=student,
        max_iterations=2,
        data_dir=str(data_dir),
        mock=True,
    )
    assert success is True
    assert record_ft["config"]["tutor_prompt"] == "finetuned_socratic_xml"

    # Test socratic_ft -> pedagogical system prompt label
    success, _, record_sft = run_single_conversation(
        tutor_model=tutor_socratic_ft,
        student_model=student,
        max_iterations=2,
        data_dir=str(data_dir),
        mock=True,
    )
    assert success is True
    assert record_sft["config"]["tutor_prompt"] == "finetuned_socratic_pedagogical_xml"


def test_pipeline_passes_pedagogical_prompt_to_socratic_ft_messages(tmp_path, monkeypatch):
    from src.pipeline import load_models_from_file, run_single_conversation
    from src.prompt_generator import FINETUNED_SOCRATIC_PEDAGOGICAL_SYSTEM_PROMPT, FINETUNED_SOCRATIC_SYSTEM_PROMPT
    from scripts.run_pipeline import PROJECT_ROOT

    tutors = load_models_from_file(str(PROJECT_ROOT / "models" / "tutors.json"))
    students = load_models_from_file(str(PROJECT_ROOT / "models" / "students.json"))

    tutor_socratic_ft = tutors[7]
    student = students[2]

    captured_tutor_prompts = []

    def mock_student_call(prompt, **kwargs):
        return True, "I think 2 + 2 = 5.", 10, None

    def mock_tutor_call(prompt, **kwargs):
        captured_tutor_prompts.append(prompt)
        return True, "<reasoning>Student made arithmetic error</reasoning><state>Closing</state><response>Check again</response>", 20, None

    monkeypatch.setattr(student, "call_model", mock_student_call)
    monkeypatch.setattr(tutor_socratic_ft, "call_model", mock_tutor_call)

    data_dir = tmp_path / "data"
    success, _, _ = run_single_conversation(
        tutor_model=tutor_socratic_ft,
        student_model=student,
        max_iterations=1,
        data_dir=str(data_dir),
        mock=False,
    )

    assert success is True
    assert len(captured_tutor_prompts) == 1
    # Verify the messages array passed to socratic_ft contains the pedagogical system prompt
    first_message = captured_tutor_prompts[0][0]
    assert first_message["role"] == "system"
    assert first_message["content"] == FINETUNED_SOCRATIC_PEDAGOGICAL_SYSTEM_PROMPT
    assert "Diagnosis" in first_message["content"]
    assert "Closing" in first_message["content"]





