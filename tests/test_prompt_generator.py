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
        {"turn": 2, "speaker": "tutor", "end": 0, "response": "Sure, what is confusing?"},
    ]
    text = format_conversation_history(history)
    assert "Turn 1 (Student): Hello, I need help." in text
    assert "Turn 2 (Tutor): Sure, what is confusing?" in text


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
    assert "Quadratic Equations" in prompt_mat
    assert "A student solves x^2 = 9" in prompt_mat


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


def test_generate_tutor_prompt_default():
    prompt = generate_tutor_prompt()
    assert "SOCRATIC TUTOR" in prompt
    assert "Diagnosis" in prompt
    assert "Newton's Third Law" in prompt or "truck collides" in prompt
    assert "no conversation history" in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt


def test_generate_tutor_prompt_with_history():
    history_str = "Turn 1 (Student): I think force depends on speed."
    prompt = generate_tutor_prompt(conversation_history=history_str)
    assert history_str in prompt
    assert "no conversation history" not in prompt

    history_list = [
        {"turn": 1, "speaker": "student", "text": "Is work zero?"},
        {"turn": 2, "speaker": "tutor", "end": 0, "response": "Why do you think so?"},
    ]
    prompt_list = generate_tutor_prompt(conversation_history=history_list)
    assert "Turn 1 (Student): Is work zero?" in prompt_list
    assert "Turn 2 (Tutor): Why do you think so?" in prompt_list


def test_generate_tutor_prompt_scenario_id():
    prompt = generate_tutor_prompt(scenario_id="MAT_001")
    assert "A student solves x^2 = 9" in prompt

    prompt_custom = generate_tutor_prompt(problem="Solve 3x + 5 = 20")
    assert "Solve 3x + 5 = 20" in prompt_custom


def test_tutor_prompt_generator_class():
    from src.prompt_generator import TutorPromptGenerator

    gen = TutorPromptGenerator()
    prompt = gen.generate_tutor_prompt(scenario_id="PHY_001")
    assert "truck" in prompt.lower()
    assert "{{" not in prompt

    prompt2 = gen.generate_prompt(
        scenario_id="CS_001",
        conversation_history="Turn 1 (Student): Nested loops are always O(n^2).",
    )
    assert "Nested loops are always O(n^2)." in prompt2


def test_generate_tutor_prompt_custom_file(tmp_path):
    custom_file = tmp_path / "custom_tutor.md"
    custom_file.write_text("# CUSTOM TUTOR\nProblem: {{QUESTION}}\nHistory: {{CONVERSATION_HISTORY}}", encoding="utf-8")

    prompt = generate_tutor_prompt(problem="What is 2+2?", template_path=custom_file)
    assert "# CUSTOM TUTOR" in prompt
    assert "Problem: What is 2+2?" in prompt
    assert "History: no conversation history" in prompt


def test_generate_tutor_prompt_named_templates():
    prompt_small = generate_tutor_prompt(template_path="socratic_tutor_prompt_small.md")
    assert "SOCRATIC TUTOR" in prompt_small

    prompt_full = generate_tutor_prompt(template_path="socratic_tutor_prompt.md")
    assert "SOCRATIC TUTOR" in prompt_full


def test_generate_tutor_prompt_missing_file():
    import pytest
    with pytest.raises(FileNotFoundError, match="Tutor prompt template not found"):
        generate_tutor_prompt(template_path="non_existent_prompt_template_9999.md")


def test_build_finetuned_tutor_messages_empty():
    from src.prompt_generator import (
        FINETUNED_SOCRATIC_SYSTEM_PROMPT,
        build_finetuned_tutor_messages,
    )

    messages = build_finetuned_tutor_messages(problem="What is 15 * 4?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == FINETUNED_SOCRATIC_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is 15 * 4?"


def test_build_finetuned_tutor_messages_multiturn_preserves_xml():
    from src.prompt_generator import (
        FINETUNED_SOCRATIC_SYSTEM_PROMPT,
        build_finetuned_tutor_messages,
    )

    raw_xml_t1 = (
        "<reasoning>Student made an arithmetic error.</reasoning>\n"
        "<state>Diagnosis</state>\n"
        "<response>Can you double check your addition?</response>"
    )

    conversation = [
        {"turn": 1, "speaker": "student", "text": "I think the answer is 10."},
        {
            "turn": 2,
            "speaker": "tutor",
            "reasoning": "Student made an arithmetic error.",
            "state": "Diagnosis",
            "end": 0,
            "response": "Can you double check your addition?",
            "raw_output": raw_xml_t1,
        },
        {"turn": 3, "speaker": "student", "text": "Oh, 5 + 6 is 11!"},
    ]

    messages = build_finetuned_tutor_messages(conversation_history=conversation)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == FINETUNED_SOCRATIC_SYSTEM_PROMPT

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "I think the answer is 10."

    # Verify full raw XML is preserved for tutor assistant history
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == raw_xml_t1
    assert "<reasoning>" in messages[2]["content"]
    assert "<state>Diagnosis</state>" in messages[2]["content"]
    assert "<response>" in messages[2]["content"]

    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Oh, 5 + 6 is 11!"


def test_separation_of_tutor_vs_student_history():
    """Verify tutor assistant history has full raw XML while student prompt receives ONLY clean response text."""
    from src.prompt_generator import (
        build_finetuned_tutor_messages,
        generate_student_prompt,
    )

    raw_xml = (
        "<reasoning>Student has misconception about gravity.</reasoning>\n"
        "<state>Support</state>\n"
        "<response>What causes the object to accelerate downward?</response>"
    )

    conversation = [
        {"turn": 1, "speaker": "student", "text": "Heavier objects fall faster."},
        {
            "turn": 2,
            "speaker": "tutor",
            "reasoning": "Student has misconception about gravity.",
            "state": "Support",
            "end": 0,
            "response": "What causes the object to accelerate downward?",
            "raw_output": raw_xml,
        },
    ]

    # 1. Tutor messages contain full raw XML
    tutor_messages = build_finetuned_tutor_messages(conversation_history=conversation)
    assert tutor_messages[2]["role"] == "assistant"
    assert tutor_messages[2]["content"] == raw_xml
    assert "<reasoning>" in tutor_messages[2]["content"]
    assert "<state>Support</state>" in tutor_messages[2]["content"]

    # 2. Student prompt receives ONLY clean response text
    student_prompt = generate_student_prompt(conversation_history=conversation)
    assert "What causes the object to accelerate downward?" in student_prompt
    assert "<reasoning>" not in student_prompt
    assert "<state>" not in student_prompt
    assert "<response>" not in student_prompt
    assert "Student has misconception about gravity" not in student_prompt


def test_tutor_prompt_generator_class_finetuned():
    from src.prompt_generator import TutorPromptGenerator

    gen = TutorPromptGenerator()
    conversation = [
        {"turn": 1, "speaker": "student", "text": "Hello, need help with calculus."}
    ]
    messages = gen.generate_finetuned_messages(
        conversation_history=conversation,
        problem="Find derivative of x^2",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "You are a Socratic tutor." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, need help with calculus."
