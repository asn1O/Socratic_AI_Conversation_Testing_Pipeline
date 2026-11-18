"""Prompt generator for student-impersonating and tutor models."""

import json
from pathlib import Path
import random
import re
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
    randomize: bool = False,
) -> Dict[str, Any]:
    """Retrieve a question scenario by scenario_id or index from question_bank.json."""
    bank = question_bank if question_bank is not None else load_question_bank(question_bank_path)
    if not bank:
        return {}

    if scenario_id:
        target_id = scenario_id.strip().upper()
        if target_id in ("RANDOM", "ANY"):
            return random.choice(bank)
        for q in bank:
            if q.get("scenario_id", "").strip().upper() == target_id:
                return q

    if index is not None and 0 <= index < len(bank):
        return bank[index]

    if randomize:
        return random.choice(bank)

    return bank[0]


def get_random_question_scenario(
    question_bank: Optional[List[Dict[str, Any]]] = None,
    question_bank_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Return a randomly chosen scenario from question_bank.json."""
    bank = question_bank if question_bank is not None else load_question_bank(question_bank_path)
    if not bank:
        return {}
    return random.choice(bank)


def get_student_persona(
    persona_id: Optional[str] = None,
    persona_data: Optional[Dict[str, Any]] = None,
    randomize: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Retrieve a student persona by ID, matching name, or return custom/random/default persona."""
    if persona_data:
        p_id = persona_id or persona_data.get("persona_id", "CUSTOM")
        return p_id, persona_data

    if persona_id:
        norm_id = persona_id.strip().upper()
        if norm_id in ("RANDOM", "ANY"):
            rand_id = random.choice(list(STUDENT_PERSONAS.keys()))
            return rand_id, STUDENT_PERSONAS[rand_id]

        if norm_id in STUDENT_PERSONAS:
            return norm_id, STUDENT_PERSONAS[norm_id]

        for pid, pdata in STUDENT_PERSONAS.items():
            if norm_id in pid or norm_id in pdata.get("persona_name", "").upper():
                return pid, pdata

    if randomize:
        rand_id = random.choice(list(STUDENT_PERSONAS.keys()))
        return rand_id, STUDENT_PERSONAS[rand_id]

    default_id = "STU_001"
    return default_id, STUDENT_PERSONAS[default_id]


def get_random_student_persona() -> Tuple[str, Dict[str, Any]]:
    """Return a randomly chosen student persona."""
    rand_id = random.choice(list(STUDENT_PERSONAS.keys()))
    return rand_id, STUDENT_PERSONAS[rand_id]


DEFAULT_EVAL_COMBOS: List[Dict[str, Any]] = [
    {
        "combo_id": "EVAL_01",
        "scenario_id": "PHY_001",
        "persona_id": "STU_002",
        "domain": "Physics",
        "subtopic": "Newton's Third Law",
        "difficulty": "medium",
        "description": "Newton's Third Law collision force with a Confident but Incorrect student (arguing truck hits car with more force).",
    },
    {
        "combo_id": "EVAL_02",
        "scenario_id": "MAT_001",
        "persona_id": "STU_004",
        "domain": "Mathematics",
        "subtopic": "Quadratic Equations",
        "difficulty": "easy",
        "description": "Quadratic Equations (x^2=9) with a Procedural Learner (only took positive root 3 through memorized procedure).",
    },
    {
        "combo_id": "EVAL_03",
        "scenario_id": "CS_001",
        "persona_id": "STU_005",
        "domain": "Computer Science",
        "subtopic": "Time Complexity and Nested Loops",
        "difficulty": "medium",
        "description": "Time complexity of nested loops with a Partially Knowledgeable Learner (assuming two loops always mean O(n^2)).",
    },
    {
        "combo_id": "EVAL_04",
        "scenario_id": "CHEM_003",
        "persona_id": "STU_001",
        "domain": "Chemistry",
        "subtopic": "Acids, Bases and pH",
        "difficulty": "easy",
        "description": "Acid dilution and pH with a Confused Beginner (confusing total volume increase with acidity increase).",
    },
    {
        "combo_id": "EVAL_05",
        "scenario_id": "LOG_001",
        "persona_id": "STU_003",
        "domain": "Logical Reasoning",
        "subtopic": "Conditional Statements and Logical Fallacies",
        "difficulty": "medium",
        "description": "Affirming the consequent with a Passive Learner (needs proactive guidance to recognize alternative causes for wet roads).",
    },
    {
        "combo_id": "EVAL_06",
        "scenario_id": "PHY_002",
        "persona_id": "STU_001",
        "domain": "Physics",
        "subtopic": "Work and Energy",
        "difficulty": "easy",
        "description": "Mechanical work on horizontally carried bag with a Confused Beginner (struggles to separate physical tiredness from mechanical work W=Fd cos(theta)).",
    },
    {
        "combo_id": "EVAL_07",
        "scenario_id": "MAT_002",
        "persona_id": "STU_002",
        "domain": "Mathematics",
        "subtopic": "Probability and Independent Events",
        "difficulty": "medium",
        "description": "Gambler's fallacy after 5 consecutive coin heads with a Confident but Incorrect student (insisting tails is due).",
    },
    {
        "combo_id": "EVAL_08",
        "scenario_id": "CS_002",
        "persona_id": "STU_004",
        "domain": "Computer Science",
        "subtopic": "Operating Systems and Virtual Memory",
        "difficulty": "hard",
        "description": "Virtual memory addressing and page faults with a Procedural Learner (believes valid virtual address must always physically reside in RAM).",
    },
    {
        "combo_id": "EVAL_09",
        "scenario_id": "CHEM_001",
        "persona_id": "STU_005",
        "domain": "Chemistry",
        "subtopic": "Periodic Trends: Atomic Radius",
        "difficulty": "medium",
        "description": "Periodic trend of atomic radius across a period with a Partially Knowledgeable Learner (overlooks effective nuclear charge and shielding).",
    },
    {
        "combo_id": "EVAL_10",
        "scenario_id": "LOG_002",
        "persona_id": "STU_003",
        "domain": "Logical Reasoning",
        "subtopic": "Boolean Logic and De Morgan's Laws",
        "difficulty": "medium",
        "description": "Negation of compound condition (A AND B) with a Passive Learner (needs step-by-step Socratic questions to discover De Morgan's laws).",
    },
]


def load_eval_combos(path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Load evaluation parameter combinations from a JSON file.
    Falls back to DEFAULT_EVAL_COMBOS if file is not found or invalid.
    """
    primary = Path(path) if path else REPO_ROOT / "eval_combos.json"
    if primary.is_file():
        try:
            with primary.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass

    return [dict(c) for c in DEFAULT_EVAL_COMBOS]


def get_eval_combo(
    index: int,
    one_indexed: bool = True,
    combos: Optional[List[Dict[str, Any]]] = None,
    combos_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Retrieve parameter combo sequentially using deterministic modulo indexing.

    If one_indexed is True (e.g. conversation index i starts from 1):
      idx = (index - 1) % len(combos)
      (i.e. i=1 -> 1st combo, i=10 -> 10th combo, i=11 -> 1st combo, i=12 -> 2nd combo)
    If one_indexed is False (0-indexed):
      idx = index % len(combos)
    """
    pool = combos if combos is not None else load_eval_combos(combos_path)
    if not pool:
        raise ValueError("Eval combos list is empty.")

    if one_indexed:
        target_idx = (index - 1) % len(pool)
    else:
        target_idx = index % len(pool)

    return pool[target_idx]


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
    """Read a markdown prompt template from the file system or repo root."""
    for fn in [filename, fallback_filename]:
        if not fn:
            continue
        p = Path(fn)
        if p.is_file():
            content = p.read_text(encoding="utf-8").strip()
            if content:
                return content
        repo_p = REPO_ROOT / fn
        if repo_p.is_file():
            content = repo_p.read_text(encoding="utf-8").strip()
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
            response = item.get("response", item.get("text", ""))
            # Ensure no XML tags or reasoning/state leak into student conversation history
            if isinstance(response, str):
                if "<response>" in response and "</response>" in response:
                    m = re.search(r"<response>(.*?)</response>", response, re.DOTALL | re.IGNORECASE)
                    if m:
                        response = m.group(1).strip()
                response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL | re.IGNORECASE)
                response = re.sub(r"<state>.*?</state>", "", response, flags=re.DOTALL | re.IGNORECASE)
                response = response.strip()
            lines.append(f"Turn {turn} (Tutor): {response}")
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


def generate_tutor_prompt(
    conversation_history: Union[str, List[Dict[str, Any]], None] = "",
    scenario_id: Optional[str] = None,
    question_data: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
    template_path: Optional[Union[str, Path]] = None,
    question_bank: Optional[List[Dict[str, Any]]] = None,
    question_bank_path: Optional[Union[str, Path]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    definitions: Optional[Dict[str, Any]] = None,
    problem: Optional[str] = None,
    question: Optional[str] = None,
    topic: Optional[str] = None,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    learning_objective: Optional[str] = None,
    target_misconception: Optional[str] = None,
    empty_history_text: str = "no conversation history",
    **kwargs: Any,
) -> str:
    """
    Generate the prompt for the tutor model.

    Defaults to socratic_tutor_prompt_small.md and populates placeholders:
      - {{QUESTION}} / {{PROBLEM}}
      - {{CONVERSATION_HISTORY}}
      - Optional contextual fields ({{DOMAIN}}, {{SUBTOPIC}}, {{DIFFICULTY}}, etc.)
    """
    # If the first positional argument was passed as a prompt template containing template markers
    if (
        isinstance(conversation_history, str)
        and prompt_template is None
        and ("# SOCRATIC TUTOR" in conversation_history or "# BASE TUTOR" in conversation_history or "{{" in conversation_history)
    ):
        prompt_template = conversation_history
        conversation_history = ""

    # Load prompt template
    if prompt_template:
        template = prompt_template
    elif template_path:
        t_path = Path(template_path)
        if not t_path.is_file() and (REPO_ROOT / template_path).is_file():
            t_path = REPO_ROOT / template_path
        if t_path.is_file():
            template = t_path.read_text(encoding="utf-8").strip()
        else:
            raise FileNotFoundError(f"Tutor prompt template not found: {template_path}")
    else:
        template = _read_prompt_template(
            "socratic_tutor_prompt_small.md",
            fallback_filename="socratic_tutor_prompt.md",
        )

    if not template.strip():
        # Fallback template matching socratic_tutor_prompt_small.md
        template = (
            "# SOCRATIC TUTOR — SYSTEM PROMPT\n\n"
            "You are an expert Socratic tutor. Your goal is to help students develop conceptual understanding "
            "through questioning, hints and reflection rather than simply providing answers.\n\n"
            "## OUTPUT FORMAT\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "reasoning": "Brief pedagogical rationale for the selected approach.",\n'
            '  "end": 0,\n'
            '  "response": "The actual message shown to the student."\n'
            "}\n\n"
            "# TUTOR CONTEXT\n\n"
            "## 1. PROBLEM STATEMENT\n\n"
            "{{QUESTION}}\n\n"
            "---\n\n"
            "## 2. CONVERSATION HISTORY\n\n"
            "{{CONVERSATION_HISTORY}}\n\n"
            "---\n\n"
            "Generate your next tutor response based on the system instructions and the conversation context provided."
        )

    # Resolve question scenario from question_bank.json
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
    q_text = question or problem or q_scenario.get("question", "")
    learn_obj = learning_objective or q_scenario.get("learning_objective", "")
    misconc = target_misconception or q_scenario.get("target_misconception", "")

    # Resolve conversation history
    conv_input = conversation_history
    if conv_input == "" and parameters and "conversation_history" in parameters:
        conv_input = parameters["conversation_history"]

    history_str = normalize_conversation_history(
        conv_input, empty_placeholder=empty_history_text
    )

    # Perform structured placeholder replacements
    replacements: Dict[str, str] = {
        "QUESTION": q_text,
        "PROBLEM": q_text,
        "CONVERSATION_HISTORY": history_str,
        "PASTE THE ENTIRE CONVERSATION HERE": history_str,
        "SCENARIO_ID": scen_id,
        "DOMAIN": dom,
        "SUBTOPIC": sub,
        "TOPIC": sub,
        "DIFFICULTY": diff,
        "LEARNING_OBJECTIVE": learn_obj,
        "TARGET_MISCONCEPTION": misconc,
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


FINETUNED_SOCRATIC_SYSTEM_PROMPT = (
    "You are a Socratic tutor. Guide the student toward the answer without giving it away. "
    "For every reply, output <reasoning>...</reasoning>, <state>...</state>, and <response>...</response>."
)


def build_finetuned_tutor_messages(
    conversation_history: Optional[Union[List[Dict[str, Any]], str]] = None,
    problem: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> List[Dict[str, str]]:
    """
    Build a standard OpenAI-compatible multi-turn messages array for the fine-tuned Socratic tutor:
      - {"role": "system", "content": <exact SFT system prompt>}
      - {"role": "user", "content": <initial problem statement / student opening message>}
      - {"role": "assistant", "content": <tutor's FULL raw XML output from turn 1>}
      - {"role": "user", "content": <student's reply from turn 1>}
      - ...and so on.

    Preserves full raw XML string (<reasoning>...</reasoning>\n<state>...</state>\n<response>...</response>)
    in the tutor's assistant history.
    """
    sys_content = (system_prompt or FINETUNED_SOCRATIC_SYSTEM_PROMPT).strip()
    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_content}]

    if not conversation_history:
        if problem:
            messages.append({"role": "user", "content": str(problem).strip()})
        return messages

    if isinstance(conversation_history, str):
        cleaned = conversation_history.strip()
        if cleaned:
            messages.append({"role": "user", "content": cleaned})
        elif problem:
            messages.append({"role": "user", "content": str(problem).strip()})
        return messages

    if isinstance(conversation_history, list):
        if len(conversation_history) == 0:
            if problem:
                messages.append({"role": "user", "content": str(problem).strip()})
            return messages

        # If conversation starts with a tutor turn and problem is provided, insert problem as initial user turn
        first_speaker = str(conversation_history[0].get("speaker", "")).strip().lower()
        if first_speaker in ("tutor", "assistant") and problem:
            messages.append({"role": "user", "content": str(problem).strip()})

        for item in conversation_history:
            speaker = str(item.get("speaker", "")).strip().lower()
            if speaker in ("student", "user"):
                text = item.get("text") or item.get("content") or item.get("response", "")
                messages.append({"role": "user", "content": str(text).strip()})
            elif speaker in ("tutor", "assistant"):
                raw_xml = item.get("raw_output") or item.get("raw_xml")
                if not raw_xml:
                    reasoning = item.get("reasoning", "")
                    state = item.get("state", "")
                    resp = item.get("response", item.get("text", ""))
                    raw_xml = f"<reasoning>{reasoning}</reasoning>\n<state>{state}</state>\n<response>{resp}</response>"
                messages.append({"role": "assistant", "content": str(raw_xml).strip()})

    return messages


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


class TutorPromptGenerator:
    """Class-based prompt generator for Socratic tutor simulator."""

    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        question_bank_path: Optional[Union[str, Path]] = None,
    ):
        if template_path:
            t_path = Path(template_path)
            if not t_path.is_file() and (REPO_ROOT / template_path).is_file():
                t_path = REPO_ROOT / template_path
            self.template_path = t_path
        else:
            self.template_path = REPO_ROOT / "socratic_tutor_prompt_small.md"
        self.question_bank_path = (
            Path(question_bank_path)
            if question_bank_path
            else REPO_ROOT / "question_bank.json"
        )
        self.question_bank = load_question_bank(self.question_bank_path)

    def generate_tutor_prompt(
        self,
        conversation_history: Union[str, List[Dict[str, Any]], None] = "",
        scenario_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate the complete tutor prompt."""
        return generate_tutor_prompt(
            conversation_history=conversation_history,
            scenario_id=scenario_id,
            template_path=self.template_path,
            question_bank=self.question_bank,
            question_bank_path=self.question_bank_path,
            **kwargs,
        )

    def generate_prompt(
        self,
        conversation_history: Union[str, List[Dict[str, Any]], None] = "",
        scenario_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Alias for generate_tutor_prompt."""
        return self.generate_tutor_prompt(
            conversation_history=conversation_history,
            scenario_id=scenario_id,
            **kwargs,
        )

    def generate_finetuned_messages(
        self,
        conversation_history: Optional[Union[List[Dict[str, Any]], str]] = None,
        problem: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Generate multi-turn messages array for fine-tuned Socratic tutor."""
        return build_finetuned_tutor_messages(
            conversation_history=conversation_history,
            problem=problem,
            system_prompt=system_prompt,
            **kwargs,
        )


DatasetPromptGenerator = StudentPromptGenerator


def main() -> None:
    """Quick preview and sanity check of generated prompts."""
    stu_generator = StudentPromptGenerator()
    print("=== Default Student Prompt (STU_001, empty conversation history) ===")
    print(stu_generator.generate_student_prompt())
    print("\n" + "=" * 60 + "\n")

    tutor_generator = TutorPromptGenerator()
    print("=== Default Tutor Prompt (socratic_tutor_prompt_small.md, empty history) ===")
    print(tutor_generator.generate_tutor_prompt())
    print("\n" + "=" * 60 + "\n")
    print("=== Tutor Prompt with Turn 1 History and MAT_001 ===")
    print(
        tutor_generator.generate_tutor_prompt(
            scenario_id="MAT_001",
            conversation_history="Turn 1 (Student): I think the answer is x = 3 because 3 squared is 9.",
        )
    )


if __name__ == "__main__":
    main()
