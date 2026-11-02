"""Prompt generator for student-impersonating and tutor models."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent

STUDENT_PERSONAS = {
    "STU_001": {
        "persona_name": "Confused Beginner",

        "definition": (
            "The student has limited prior knowledge of the subject "
            "and struggles to understand the underlying concepts. "
            "They may recognize basic terminology but have difficulty "
            "connecting ideas or explaining their reasoning. They are "
            "often uncertain about their answers and require questions "
            "and guidance to develop their understanding."
        ),

        "behavioral_guidelines": [
            "Demonstrate limited initial understanding of the topic.",
            "Express uncertainty when unsure about an answer.",
            "Struggle to articulate reasoning or connect concepts.",
            "Frequently require clarification or simpler explanations.",
            "Make gradual progress when the tutor provides appropriate guidance.",
            "Do not immediately understand complex hints or explanations."
        ]
    },

    "STU_002": {
        "persona_name": "Confident but Incorrect",

        "definition": (
            "The student has a moderate understanding of the subject "
            "but holds a specific misconception related to the problem. "
            "They are confident in their existing understanding and "
            "initially defend their reasoning when challenged. They "
            "do not readily accept explanations that contradict their "
            "beliefs. However, they are willing to reconsider their "
            "understanding when the tutor provides convincing reasoning "
            "and asks appropriate questions."
        ),

        "behavioral_guidelines": [
            "Express confidence when presenting existing beliefs.",
            "Initially demonstrate the target misconception when applicable.",
            "Attempt to justify incorrect reasoning when challenged.",
            "Do not immediately accept corrections from the tutor.",
            "Reconsider beliefs when presented with convincing reasoning.",
            "Gradually demonstrate corrected understanding when appropriate."
        ]
    },

    "STU_003": {
        "persona_name": "Passive Learner",

        "definition": (
            "The student has limited prior knowledge and tends to take "
            "a passive role in the learning process. They generally "
            "wait for the tutor to initiate questions and provide "
            "guidance rather than actively exploring the problem "
            "themselves. Their responses are usually brief and they "
            "rarely elaborate unless explicitly encouraged."
        ),

        "behavioral_guidelines": [
            "Give relatively short and straightforward responses.",
            "Avoid spontaneously providing lengthy explanations.",
            "Rarely initiate questions or introduce new reasoning.",
            "Respond primarily to what the tutor explicitly asks.",
            "Require encouragement to elaborate on answers.",
            "Show gradual understanding when actively guided."
        ]
    },

    "STU_004": {
        "persona_name": "Procedural Learner",

        "definition": (
            "The student can follow procedures, remember formulas "
            "and perform individual steps but struggles to understand "
            "the underlying concepts and why those procedures work. "
            "They may arrive at correct answers through memorized "
            "methods without being able to explain the reasoning "
            "behind them or generalize the knowledge to unfamiliar "
            "situations."
        ),

        "behavioral_guidelines": [
            "Demonstrate familiarity with relevant formulas or procedures.",
            "Attempt to solve problems by recalling known steps.",
            "Provide correct intermediate results without necessarily understanding why.",
            "Struggle when asked to explain the underlying concept.",
            "Have difficulty generalizing procedures to unfamiliar situations.",
            "Develop conceptual understanding when guided beyond procedural steps."
        ]
    },

    "STU_005": {
        "persona_name": "Partially Knowledgeable Learner",

        "definition": (
            "The student has a reasonable understanding of the "
            "fundamentals and can independently reason about parts "
            "of the problem. However, their understanding contains "
            "gaps and they may make incorrect assumptions or overlook "
            "important details. They are generally receptive to "
            "guidance and can often make meaningful progress after "
            "a small number of well-directed questions or hints."
        ),

        "behavioral_guidelines": [
            "Demonstrate reasonable initial knowledge of the topic.",
            "Attempt independent reasoning before seeking help.",
            "Occasionally make incorrect assumptions or overlook details.",
            "Provide explanations for answers rather than simply stating them.",
            "Respond constructively to hints and targeted questions.",
            "Connect new explanations with previously understood concepts.",
            "Demonstrate the ability to generalize once gaps are resolved."
        ]
    }
}


def load_question_bank(path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load question scenarios from question_bank.json."""
    primary_path = Path(path) if path else REPO_ROOT / "question_bank.json"
    if primary_path.is_file():
        with primary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get("questions", [])
            elif isinstance(data, list):
                return data

    fallback = Path(__file__).resolve().parent.parent / "question_bank.json"
    if fallback.is_file():
        with fallback.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get("questions", [])
            elif isinstance(data, list):
                return data

    return []


def get_question_scenario(
    scenario_id: Optional[str] = None,
    question_bank: Optional[List[Dict[str, Any]]] = None,
    question_bank_path: Optional[Union[str, Path]] = None,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieve a question scenario by scenario_id or index from question_bank.json."""
    bank = question_bank if question_bank is not None else load_question_bank(question_bank_path)
    if not bank:
        return {}

    if scenario_id:
        target_id = scenario_id.strip().upper()
        for q in bank:
            if q.get("scenario_id", "").strip().upper() == target_id:
                return q

    if index is not None and 0 <= index < len(bank):
        return bank[index]

    return bank[0]


def get_student_persona(
    persona_id: Optional[str] = None,
    persona_data: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Retrieve a student persona by ID, matching name, or return custom/default persona."""
    if persona_data:
        p_id = persona_id or persona_data.get("persona_id", "CUSTOM")
        return p_id, persona_data

    if persona_id:
        norm_id = persona_id.strip().upper()
        if norm_id in STUDENT_PERSONAS:
            return norm_id, STUDENT_PERSONAS[norm_id]

        for pid, pdata in STUDENT_PERSONAS.items():
            if norm_id in pid or norm_id in pdata.get("persona_name", "").upper():
                return pid, pdata

    default_id = "STU_001"
    return default_id, STUDENT_PERSONAS[default_id]


def format_behavioral_guidelines(guidelines: Union[List[str], str, None]) -> str:
    """Format behavioural guidelines as markdown bullet points."""
    if not guidelines:
        return ""
    if isinstance(guidelines, str):
        return guidelines.strip()
    formatted = []
    for g in guidelines:
        g_str = str(g).strip()
        if not g_str.startswith("-"):
            formatted.append(f"- {g_str}")
        else:
            formatted.append(g_str)
    return "\n".join(formatted)


def normalize_conversation_history(
    conversation_history: Union[str, List[Dict[str, Any]], None],
    empty_placeholder: str = "no conversation history",
) -> str:
    """
    Process conversation history into a clean string representation.
    If the string or list is empty or whitespace-only, returns the empty_placeholder.
    """
    if conversation_history is None:
        return empty_placeholder

    if isinstance(conversation_history, str):
        cleaned = conversation_history.strip()
        return cleaned if cleaned else empty_placeholder

    if isinstance(conversation_history, list):
        if not conversation_history:
            return empty_placeholder
        return format_conversation_history(conversation_history)

    str_val = str(conversation_history).strip()
    return str_val if str_val else empty_placeholder



def _read_prompt_template(filename: str, fallback_filename: Optional[str] = None) -> str:
    """Read a markdown prompt template from the repo root."""
    primary_path = REPO_ROOT / filename
    if primary_path.is_file():
        content = primary_path.read_text(encoding="utf-8").strip()
        if content:
            return content

    if fallback_filename:
        fallback_path = REPO_ROOT / fallback_filename
        if fallback_path.is_file():
            content = fallback_path.read_text(encoding="utf-8").strip()
            if content:
                return content

    return ""


def format_conversation_history(
    conversation: Optional[List[Dict[str, Any]]],
    style: str = "transcript",
) -> str:
    """Format conversation turns into a clean representation for prompt injection."""
    if not conversation:
        return "No conversation history yet. This is the first turn."

    if style == "json":
        return json.dumps(conversation, indent=2, ensure_ascii=False)

    lines = []
    for item in conversation:
        turn = item.get("turn", "?")
        speaker = item.get("speaker", "unknown")
        if speaker == "tutor":
            state = item.get("state", "")
            response = item.get("response", item.get("text", ""))
            state_str = f" [{state}]" if state else ""
            lines.append(f"Turn {turn} (Tutor{state_str}): {response}")
        else:
            text = item.get("text", "")
            lines.append(f"Turn {turn} (Student): {text}")

    return "\n\n".join(lines)


def inject_variables(
    base_text: str,
    conversation_history: Optional[Union[List[Dict[str, Any]], str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    definitions: Optional[Dict[str, Any]] = None,
    problem: Optional[str] = None,
    topic: Optional[str] = None,
    domain: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """
    Inject parameters, definitions, conversation history, and contextual fields
    into prompt text. Supports placeholder styles:
      - {{VARIABLE}}
      - <<VARIABLE>>
      - [VARIABLE]
      - insert_variable
    """
    history_text = (
        format_conversation_history(conversation_history)
        if isinstance(conversation_history, list)
        else normalize_conversation_history(conversation_history)
    )
    history_json = (
        json.dumps(conversation_history, indent=2, ensure_ascii=False)
        if isinstance(conversation_history, list)
        else history_text
    )

    replacements: Dict[str, str] = {
        "CONVERSATION_HISTORY": history_text,
        "CONVERSATION_JSON": history_json,
        "PASTE THE ENTIRE CONVERSATION HERE": history_text,
        "PROBLEM": problem or "",
        "QUESTION": problem or "",
        "TOPIC": topic or "",
        "SUBTOPIC": topic or "",
        "DOMAIN": domain or "",
    }

    if parameters:
        for k, v in parameters.items():
            replacements[k.upper()] = str(v)
            replacements[f"insert_{k.lower()}"] = str(v)

    if definitions:
        for k, v in definitions.items():
            replacements[f"{k.upper()}_DEF"] = str(v)
            replacements[f"insert_{k.lower()}_def"] = str(v)

    for k, v in kwargs.items():
        replacements[k.upper()] = str(v)
        replacements[f"insert_{k.lower()}"] = str(v)

    result = base_text
    for key, value in replacements.items():
        result = result.replace(f"{{{{{key}}}}}", value)
        result = result.replace(f"{{{{{key.lower()}}}}}", value)
        result = result.replace(f"<<{key}>>", value)
        result = result.replace(f"<<{key.lower()}>>", value)
        result = result.replace(f"[{key}]", value)

    return result


def generate_student_prompt(
    conversation_history: Union[str, List[Dict[str, Any]], None] = "",
    persona_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
    question_data: Optional[Dict[str, Any]] = None,
    persona_data: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
    template_path: Optional[Union[str, Path]] = None,
    question_bank: Optional[List[Dict[str, Any]]] = None,
    question_bank_path: Optional[Union[str, Path]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    definitions: Optional[Dict[str, Any]] = None,
    problem: Optional[str] = None,
    topic: Optional[str] = None,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    learning_objective: Optional[str] = None,
    target_misconception: Optional[str] = None,
    empty_history_text: str = "no conversation history",
    **kwargs: Any,
) -> str:
    """
    Generate the prompt for the student simulator model.

    Populates all placeholders in student_prompt.md:
      - {{PERSONA_ID}}, {{PERSONA_NAME}}, {{PERSONA_DEFINITION}}, {{BEHAVIORAL_GUIDELINES}}
      - {{SCENARIO_ID}}, {{DOMAIN}}, {{SUBTOPIC}}, {{DIFFICULTY}}, {{QUESTION}},
        {{LEARNING_OBJECTIVE}}, {{TARGET_MISCONCEPTION}}
      - {{CONVERSATION_HISTORY}} (string input, defaults to empty_history_text if empty)
    """
    # If the first positional argument was passed as a prompt template containing template markers
    if (
        isinstance(conversation_history, str)
        and prompt_template is None
        and ("# STUDENT" in conversation_history or "{{" in conversation_history)
    ):
        prompt_template = conversation_history
        conversation_history = ""

    # Load prompt template
    if prompt_template:
        template = prompt_template
    elif template_path:
        template = Path(template_path).read_text(encoding="utf-8").strip()
    else:
        template = _read_prompt_template("student_prompt.md")

    if not template.strip():
        # Fallback template if student_prompt.md is not available
        template = (
            "# STUDENT SIMULATOR\n\n"
            "# STUDENT CONTEXT\n\n"
            "## 1. STUDENT PERSONA\n\n"
            "- persona_id: {{PERSONA_ID}}\n"
            "- persona_name: {{PERSONA_NAME}}\n\n"
            "### Persona Definition\n"
            "{{PERSONA_DEFINITION}}\n\n"
            "### Behavioural Guidelines\n"
            "{{BEHAVIORAL_GUIDELINES}}\n\n"
            "---\n\n"
            "## 2. QUESTION AND PROBLEM CONTEXT\n\n"
            "- scenario_id: {{SCENARIO_ID}}\n"
            "- domain: {{DOMAIN}}\n"
            "- subtopic: {{SUBTOPIC}}\n"
            "- difficulty: {{DIFFICULTY}}\n\n"
            "### Problem Statement\n"
            "{{QUESTION}}\n\n"
            "### Learning Objective\n"
            "{{LEARNING_OBJECTIVE}}\n\n"
            "### Target Misconception\n"
            "{{TARGET_MISCONCEPTION}}\n\n"
            "---\n\n"
            "## 3. CONVERSATION HISTORY\n\n"
            "{{CONVERSATION_HISTORY}}\n\n"
            "---\n\n"
            "Generate your next student dialogue message based on the instructions and context provided."
        )

    # 1. Resolve student persona
    selected_persona_id = (
        persona_id
        or (parameters.get("persona_id") if parameters else None)
        or kwargs.get("persona_id")
        or kwargs.get("persona")
    )
    resolved_persona_id, resolved_persona = get_student_persona(
        persona_id=selected_persona_id,
        persona_data=persona_data,
    )
    p_name = resolved_persona.get("persona_name", resolved_persona_id)
    p_def = resolved_persona.get("definition", "")
    p_guidelines = format_behavioral_guidelines(
        resolved_persona.get("behavioral_guidelines", [])
    )

    # 2. Resolve question scenario from question_bank.json
    selected_scenario_id = (
        scenario_id
        or (parameters.get("scenario_id") if parameters else None)
        or kwargs.get("scenario_id")
    )
    q_scenario = (
        question_data
        if question_data
        else get_question_scenario(
            scenario_id=selected_scenario_id,
            question_bank=question_bank,
            question_bank_path=question_bank_path,
        )
    )

    scen_id = selected_scenario_id or q_scenario.get("scenario_id", "SCEN_001")
    dom = domain or q_scenario.get("domain", "")
    sub = topic or q_scenario.get("subtopic", "")
    diff = difficulty or q_scenario.get("difficulty", "")
    q_text = problem or q_scenario.get("question", "")
    learn_obj = learning_objective or q_scenario.get("learning_objective", "")
    misconc = target_misconception or q_scenario.get("target_misconception", "")

    # 3. Resolve conversation history
    conv_input = conversation_history
    if conv_input == "" and parameters and "conversation_history" in parameters:
        conv_input = parameters["conversation_history"]

    history_str = normalize_conversation_history(
        conv_input, empty_placeholder=empty_history_text
    )

    # 4. Perform structured placeholder replacements
    replacements: Dict[str, str] = {
        "PERSONA_ID": resolved_persona_id,
        "PERSONA_NAME": p_name,
        "PERSONA_DEFINITION": p_def,
        "BEHAVIORAL_GUIDELINES": p_guidelines,
        "BEHAVIOURAL_GUIDELINES": p_guidelines,
        "SCENARIO_ID": scen_id,
        "DOMAIN": dom,
        "SUBTOPIC": sub,
        "TOPIC": sub,
        "DIFFICULTY": diff,
        "QUESTION": q_text,
        "PROBLEM": q_text,
        "LEARNING_OBJECTIVE": learn_obj,
        "TARGET_MISCONCEPTION": misconc,
        "CONVERSATION_HISTORY": history_str,
        "PASTE THE ENTIRE CONVERSATION HERE": history_str,
    }

    if parameters:
        for k, v in parameters.items():
            replacements[k.upper()] = str(v)
            replacements[f"insert_{k.lower()}"] = str(v)

    if definitions:
        for k, v in definitions.items():
            replacements[f"{k.upper()}_DEF"] = str(v)
            replacements[f"insert_{k.lower()}_def"] = str(v)

    for k, v in kwargs.items():
        replacements[k.upper()] = str(v)
        replacements[f"insert_{k.lower()}"] = str(v)

    result = template
    for key, value in replacements.items():
        result = result.replace(f"{{{{{key}}}}}", value)
        result = result.replace(f"{{{{{key.lower()}}}}}", value)
        result = result.replace(f"<<{key}>>", value)
        result = result.replace(f"<<{key.lower()}>>", value)
        result = result.replace(f"[{key}]", value)

    return result


def generate_tutor_prompt(*args: Any, **kwargs: Any) -> str:
    """Tutor prompt generator (kept empty for now as requested)."""
    # Kept empty for now as tutor prompt template is pending
    return ""


class StudentPromptGenerator:
    """Class-based prompt generator for student simulator."""

    STUDENT_PERSONAS = STUDENT_PERSONAS

    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        question_bank_path: Optional[Union[str, Path]] = None,
    ):
        self.template_path = (
            Path(template_path)
            if template_path
            else REPO_ROOT / "student_prompt.md"
        )
        self.question_bank_path = (
            Path(question_bank_path)
            if question_bank_path
            else REPO_ROOT / "question_bank.json"
        )
        self.question_bank = load_question_bank(self.question_bank_path)

    def generate_student_prompt(
        self,
        conversation_history: Union[str, List[Dict[str, Any]], None] = "",
        persona_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate the complete student prompt."""
        return generate_student_prompt(
            conversation_history=conversation_history,
            persona_id=persona_id,
            scenario_id=scenario_id,
            template_path=self.template_path,
            question_bank=self.question_bank,
            question_bank_path=self.question_bank_path,
            **kwargs,
        )

    def generate_prompt(
        self,
        conversation_history: Union[str, List[Dict[str, Any]], None] = "",
        persona_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Alias for generate_student_prompt."""
        return self.generate_student_prompt(
            conversation_history=conversation_history,
            persona_id=persona_id,
            scenario_id=scenario_id,
            **kwargs,
        )


DatasetPromptGenerator = StudentPromptGenerator


def main() -> None:
    """Quick preview and sanity check of generated prompts."""
    generator = StudentPromptGenerator()
    print("=== Default Prompt (STU_001, empty conversation history) ===")
    print(generator.generate_student_prompt())
    print("\n" + "=" * 60 + "\n")
    print("=== STU_002 with MAT_001 and non-empty history ===")
    print(
        generator.generate_student_prompt(
            persona_id="STU_002",
            scenario_id="MAT_001",
            conversation_history="Turn 1 (Tutor): What numbers squared equal 9?",
        )
    )


if __name__ == "__main__":
    main()
