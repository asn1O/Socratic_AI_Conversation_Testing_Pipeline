"""Tests for student and tutor prompt generation."""

from src.prompt_generator import (
    format_conversation_history,
    generate_student_prompt,
    generate_tutor_prompt,
    inject_variables,
)


def test_format_conversation_history_empty():
    text = format_conversation_history([])
    assert "No conversation history" in text


def test_format_conversation_history_turns():
    history = [
        {"turn": 1, "speaker": "student", "text": "Hello, I need help."},
        {"turn": 2, "speaker": "tutor", "state": "Diagnosis", "response": "Sure, what is confusing?"},
    ]
    text = format_conversation_history(history)
    assert "Turn 1 (Student): Hello, I need help." in text
    assert "Turn 2 (Tutor [Diagnosis]): Sure, what is confusing?" in text


def test_inject_variables():
    template = "Problem: <<PROBLEM>>\nLevel: <<SOCRATIC_LEVEL>>\nHistory:\n<<CONVERSATION_HISTORY>>"
    history = [{"turn": 1, "speaker": "student", "text": "Test"}]
    params = {"socratic_level": "medium"}
    result = inject_variables(
        template,
        conversation_history=history,
        parameters=params,
        problem="Find x",
    )
    assert "Problem: Find x" in result
    assert "Level: medium" in result
    assert "Turn 1 (Student): Test" in result


def test_generate_student_prompt_default():
    prompt = generate_student_prompt()
    assert "STUDENT SIMULATOR" in prompt
    assert "persona_id: STU_001" in prompt
    assert "persona_name: Confused Beginner" in prompt
    assert "scenario_id: PHY_001" in prompt
    assert "no conversation history" in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt


def test_student_prompt_empty_conversation_history():
    prompt_empty = generate_student_prompt(conversation_history="")
    assert "no conversation history" in prompt_empty

    prompt_spaces = generate_student_prompt(conversation_history="   ")
    assert "no conversation history" in prompt_spaces

    prompt_none = generate_student_prompt(conversation_history=None)
    assert "no conversation history" in prompt_none


def test_student_prompt_with_conversation_history():
    history = "Turn 1 (Student): I don't understand.\nTurn 2 (Tutor): What part is unclear?"
    prompt = generate_student_prompt(conversation_history=history)
    assert history in prompt
    assert "no conversation history" not in prompt


def test_student_personas_all():
    from src.prompt_generator import STUDENT_PERSONAS

    assert len(STUDENT_PERSONAS) == 5
    for pid, pdata in STUDENT_PERSONAS.items():
        prompt = generate_student_prompt(persona_id=pid)
        assert f"persona_id: {pid}" in prompt
        assert f"persona_name: {pdata['persona_name']}" in prompt
        assert pdata["definition"] in prompt
        for g in pdata["behavioral_guidelines"]:
            assert g in prompt


def test_question_bank_scenarios():
    prompt_phy = generate_student_prompt(scenario_id="PHY_001")
    assert "scenario_id: PHY_001" in prompt_phy
    assert "Newton's Third Law" in prompt_phy

    prompt_mat = generate_student_prompt(persona_id="STU_002", scenario_id="MAT_001")
    assert "persona_id: STU_002" in prompt_mat
    assert "scenario_id: MAT_001" in prompt_mat
    assert "Algebra and Quadratic Equations" in prompt_mat
    assert "A student solves x² = 9" in prompt_mat


def test_student_prompt_generator_class():
    from src.prompt_generator import StudentPromptGenerator, DatasetPromptGenerator

    gen = StudentPromptGenerator()
    prompt = gen.generate_student_prompt(persona_id="STU_003", scenario_id="CS_001")
    assert "persona_id: STU_003" in prompt
    assert "scenario_id: CS_001" in prompt
    assert "Time Complexity" in prompt

    # Alias check
    assert DatasetPromptGenerator is StudentPromptGenerator
    gen2 = DatasetPromptGenerator()
    prompt2 = gen2.generate_prompt(persona_id="STU_004", scenario_id="BIO_001")
    assert "persona_id: STU_004" in prompt2
    assert "scenario_id: BIO_001" in prompt2


def test_generate_tutor_prompt_empty():
    prompt = generate_tutor_prompt()
    assert prompt == ""
