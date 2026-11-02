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
