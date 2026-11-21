"""Conversation testing pipeline coordinating student and tutor models."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from .model import Model
from .prompt_generator import (
    FINETUNED_SOCRATIC_PEDAGOGICAL_SYSTEM_PROMPT,
    FINETUNED_SOCRATIC_SYSTEM_PROMPT,
    build_finetuned_tutor_messages,
    generate_student_prompt,
    generate_tutor_prompt,
    get_eval_combo,
    get_question_scenario,
    get_random_question_scenario,
    get_random_student_persona,
    get_student_persona,
    load_eval_combos,
)
from .validator import (
    validate_finetuned_tutor_turn,
    validate_student_turn,
    validate_tutor_turn,
)

logger = logging.getLogger("conversation_pipeline")
logger.setLevel(logging.INFO)


def load_models_from_file(models_path: str) -> List[Model]:
    """Load model configurations from a JSON file into a list of Model objects."""
    path = Path(models_path)
    if not path.is_file():
        raise FileNotFoundError(f"models file not found: {models_path}")

    load_dotenv()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {models_path}")

    return [Model.from_dict(entry) for entry in data]


def get_model_by_index_or_name(models: List[Model], index: Optional[int] = None, name: Optional[str] = None) -> Model:
    """Select a model by matching substring in its model_name or by index."""
    if not models:
        raise ValueError("Models list is empty")

    if name:
        target = name.strip().lower()
        # 1. Exact match on response_file_stem or model_name
        for m in models:
            if target == m.response_file_stem.lower() or target == m.model_name.lower():
                return m
        # 2. Substring in model_name (preserves existing priority)
        for m in models:
            if target in m.model_name.lower():
                return m
        # 3. Substring in provider or response_file_stem
        for m in models:
            if target in m.provider.lower() or target in m.response_file_stem.lower():
                return m
        raise ValueError(f"No model found matching name '{name}'")

    if index is not None:
        if 0 <= index < len(models):
            return models[index]
        raise IndexError(f"Model index {index} out of bounds (0 to {len(models) - 1})")

    return models[0]


def run_single_conversation(
    tutor_model: Model,
    student_model: Model,
    max_iterations: int = 15,
    max_tutor_retries: int = 5,
    max_student_retries: int = 5,
    problem: Optional[str] = None,
    topic: Optional[str] = None,
    domain: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    definitions: Optional[Dict[str, Any]] = None,
    data_dir: str = "data",
    generator_id: Optional[str] = None,
    require_closing: bool = False,
    mock: bool = False,
    log_file: Optional[str] = None,
    scenario_id: Optional[str] = None,
    persona_id: Optional[str] = None,
    tutor_prompt_path: Optional[Union[str, Path]] = None,
    eval_mode: bool = False,
    eval_combo_index: Optional[int] = None,
    eval_combos: Optional[List[Dict[str, Any]]] = None,
    eval_combos_path: Optional[Union[str, Path]] = None,
    tutor_type: Optional[str] = None,
    history_window: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[bool, Optional[Path], Dict[str, Any]]:
    """
    Run a multi-turn conversation between a student model and a tutor model.

    Flow:
      For round 0 .. max_iterations-1:
        1. Student prompt generated -> Student model called (up to max_student_retries) -> validate response.
        2. Tutor prompt generated -> Tutor model called (up to max_tutor_retries) -> validate tutor JSON.
        3. If tutor end == 1, break immediately.
      Save valid conversation to data/{tutor_stem}/{generator_id}-set[n].json.
    """
    if log_file and str(log_file).strip().lower() == "auto":
        repo_root = Path(__file__).resolve().parent.parent
        log_file = str(repo_root / "logs" / f"{tutor_model.response_file_stem}.log")

    if log_file and str(log_file).strip().lower() not in ("none", "false", ""):
        log_p = Path(log_file)
        log_p.parent.mkdir(parents=True, exist_ok=True)
        all_handlers = list(logger.handlers) + list(logging.getLogger().handlers)
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).resolve() == log_p.resolve()
            for h in all_handlers
        )
        if not has_file_handler:
            fh = logging.FileHandler(str(log_p), mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(fh)

    conversation: List[Dict[str, Any]] = []
    closed = False
    rounds_completed = 0

    active_parameters = dict(parameters or {})

    # In eval mode, resolve parameters sequentially from eval combos
    eval_combo_data = None
    if eval_mode:
        eval_idx = eval_combo_index if eval_combo_index is not None else 1
        eval_combo_data = get_eval_combo(
            eval_idx,
            one_indexed=True,
            combos=eval_combos,
            combos_path=eval_combos_path,
        )
        selected_scenario_id = eval_combo_data.get("scenario_id")
        selected_persona_id = eval_combo_data.get("persona_id")
        active_parameters["eval_mode"] = True
        active_parameters["eval_combo_id"] = eval_combo_data.get("combo_id")
        active_parameters["eval_combo_index"] = eval_idx
    else:
        selected_scenario_id = scenario_id or active_parameters.get("scenario_id")
        selected_persona_id = (
            persona_id
            or active_parameters.get("persona_id")
            or active_parameters.get("persona")
        )

    # 1. Resolve Scenario from Question Bank (random by default if not specified)
    if not selected_scenario_id or str(selected_scenario_id).strip().lower() in ("random", "any"):
        scenario_data = get_random_question_scenario()
    else:
        scenario_data = get_question_scenario(scenario_id=selected_scenario_id)

    problem_text = (
        problem
        or active_parameters.get("problem")
        or active_parameters.get("question")
        or scenario_data.get("question")
        or "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    )
    topic_text = (
        topic
        or active_parameters.get("topic")
        or active_parameters.get("subtopic")
        or scenario_data.get("subtopic")
        or "Word problems & algebra"
    )
    domain_text = (
        domain
        or active_parameters.get("domain")
        or scenario_data.get("domain")
        or "Quantitative reasoning"
    )
    difficulty_text = (
        active_parameters.get("difficulty")
        or scenario_data.get("difficulty")
        or "easy"
    )
    learning_obj = (
        active_parameters.get("learning_objective")
        or scenario_data.get("learning_objective")
        or ""
    )
    target_miscon = (
        active_parameters.get("target_misconception")
        or scenario_data.get("target_misconception")
        or ""
    )
    resolved_scenario_id = scenario_data.get("scenario_id", "CUSTOM")

    # 2. Resolve Student Persona (random by default if not specified)
    if not selected_persona_id or str(selected_persona_id).strip().lower() in ("random", "any"):
        resolved_persona_id, persona_dict = get_random_student_persona()
    else:
        resolved_persona_id, persona_dict = get_student_persona(persona_id=selected_persona_id)
    persona_name = persona_dict.get("persona_name", "Confused Beginner")

    # 3. Populate active_parameters with student persona & question bank details
    active_parameters["persona_id"] = resolved_persona_id
    active_parameters["persona_name"] = persona_name
    active_parameters["scenario_id"] = resolved_scenario_id
    active_parameters["difficulty"] = difficulty_text
    if learning_obj:
        active_parameters["learning_objective"] = learning_obj
    if target_miscon:
        active_parameters["target_misconception"] = target_miscon

    repo_root = Path(__file__).resolve().parent.parent
    if tutor_prompt_path:
        t_path = Path(tutor_prompt_path)
        if not t_path.is_file() and (repo_root / tutor_prompt_path).is_file():
            t_path = repo_root / tutor_prompt_path
        resolved_tutor_prompt_path = t_path
    elif getattr(tutor_model, "tutor_prompt", None):
        t_path = Path(tutor_model.tutor_prompt)
        if not t_path.is_file() and (repo_root / tutor_model.tutor_prompt).is_file():
            t_path = repo_root / tutor_model.tutor_prompt
        resolved_tutor_prompt_path = t_path
    else:
        resolved_tutor_prompt_path = repo_root / "socratic_tutor_prompt_small.md"

    effective_history_window = (
        history_window
        if history_window is not None
        else getattr(tutor_model, "history_window", None)
    )

    is_finetuned = (
        tutor_type == "finetuned_socratic"
        or getattr(tutor_model, "is_finetuned_socratic", False)
    )

    resolved_system_prompt = None
    if is_finetuned:
        sys_choice = (
            system_prompt
            or getattr(tutor_model, "system_prompt", None)
            or ("pedagogical" if tutor_model.provider == "socratic_ft" else None)
        )
        if sys_choice in ("pedagogical", "small", "socratic", "socratic_small", "socratic_ft"):
            resolved_system_prompt = FINETUNED_SOCRATIC_PEDAGOGICAL_SYSTEM_PROMPT
            tutor_prompt_label = "finetuned_socratic_pedagogical_xml"
        elif sys_choice in ("default", "base", "ft", "finetuned_socratic"):
            resolved_system_prompt = FINETUNED_SOCRATIC_SYSTEM_PROMPT
            tutor_prompt_label = "finetuned_socratic_xml"
        elif sys_choice and len(str(sys_choice).strip()) > 30:
            resolved_system_prompt = str(sys_choice).strip()
            tutor_prompt_label = "finetuned_socratic_custom_xml"
        else:
            resolved_system_prompt = FINETUNED_SOCRATIC_SYSTEM_PROMPT
            tutor_prompt_label = "finetuned_socratic_xml"
    else:
        tutor_prompt_label = resolved_tutor_prompt_path.name

    window_label = f", Window={effective_history_window}" if effective_history_window else ""
    logger.info(
        "Starting conversation test: Tutor=%s (%s, Type=%s%s), TutorPrompt=%s, Student=%s (%s, Persona=%s/%s, Scenario=%s)",
        tutor_model.model_name,
        tutor_model.provider,
        "finetuned_socratic" if is_finetuned else "base",
        window_label,
        tutor_prompt_label,
        student_model.model_name,
        student_model.provider,
        resolved_persona_id,
        persona_name,
        resolved_scenario_id,
    )

    for i in range(max_iterations):
        rounds_completed = i + 1
        turn_student = 2 * i + 1
        turn_tutor = 2 * i + 2

        # -------------------------------------------------------------
        # 1. Student Turn (with retry loop)
        # -------------------------------------------------------------
        student_prompt = generate_student_prompt(
            conversation_history=conversation,
            persona_id=resolved_persona_id,
            scenario_id=resolved_scenario_id,
            parameters=active_parameters,
            definitions=definitions,
            problem=problem_text,
            topic=topic_text,
            domain=domain_text,
            difficulty=difficulty_text,
            learning_objective=learning_obj,
            target_misconception=target_miscon,
        )

        student_turn_valid = False
        student_reply = ""
        last_student_err = ""

        for retry in range(max_student_retries):
            logger.info(
                "Calling Student model [%s (%s)] on turn %d (attempt %d/%d)...",
                student_model.model_name,
                student_model.provider,
                turn_student,
                retry + 1,
                max_student_retries,
            )

            if mock:
                # Mock responses for testing
                if i == 0:
                    raw_student_reply = "I think the ball costs 10 cents, because $1.10 minus $1.00 is $0.10."
                elif i == 1:
                    raw_student_reply = "Wait, if the ball is 10 cents and the bat is $1 more, the bat would be $1.10, which totals $1.20! So the ball must be 5 cents."
                else:
                    raw_student_reply = "In general, when one item is X more than another, you subtract the difference from the total and divide by 2."
                student_tokens = 25
                success = True
                err = None
            else:
                success, raw_student_reply, student_tokens, err = student_model.call_model(student_prompt)

            if not success or not raw_student_reply:
                last_student_err = f"model call error: {err}"
                logger.warning(
                    "Student model [%s (%s)] call FAILED on turn %d (attempt %d/%d): error=%s",
                    student_model.model_name,
                    student_model.provider,
                    turn_student,
                    retry + 1,
                    max_student_retries,
                    err,
                )
                continue

            valid_student, student_text, student_err = validate_student_turn(raw_student_reply, active_parameters)
            if not valid_student:
                last_student_err = f"validation error: {student_err}"
                logger.warning(
                    "Student model [%s (%s)] response REJECTED on turn %d (attempt %d/%d): %s",
                    student_model.model_name,
                    student_model.provider,
                    turn_student,
                    retry + 1,
                    max_student_retries,
                    student_err,
                )
                continue

            student_turn_valid = True
            student_reply = student_text or raw_student_reply
            logger.info(
                "Student model [%s (%s)] call SUCCESS on turn %d (%d tokens):\n\"%s\"",
                student_model.model_name,
                student_model.provider,
                turn_student,
                student_tokens or 0,
                student_reply,
            )
            break

        if not student_turn_valid or not student_reply:
            logger.error(
                "Student model [%s (%s)] failed after %d retries on turn %d. Stopping pipeline.",
                student_model.model_name,
                student_model.provider,
                max_student_retries,
                turn_student,
            )
            tutor_model.write_rejected_response(
                conversation,
                data_dir=data_dir,
                generator_id=generator_id,
                reason=f"Student failed after {max_student_retries} retries on turn {turn_student}: {last_student_err}",
            )
            return False, None, {"error": f"student failed after {max_student_retries} retries on turn {turn_student}: {last_student_err}"}

        conversation.append({
            "turn": turn_student,
            "speaker": "student",
            "text": student_reply,
        })
        logger.info("Student turn %d recorded (%d words)", turn_student, len(student_reply.split()))

        # -------------------------------------------------------------
        # 2. Tutor Turn (with retry loop for valid JSON structure)
        # -------------------------------------------------------------
        tutor_turn_valid = False
        tutor_payload: Optional[Dict[str, Any]] = None
        last_tutor_err = ""

        for retry in range(max_tutor_retries):
            logger.info(
                "Calling Tutor model [%s (%s)] on turn %d (attempt %d/%d)...",
                tutor_model.model_name,
                tutor_model.provider,
                turn_tutor,
                retry + 1,
                max_tutor_retries,
            )

            if is_finetuned:
                tutor_prompt = build_finetuned_tutor_messages(
                    conversation_history=conversation,
                    problem=problem_text,
                    system_prompt=resolved_system_prompt,
                    history_window=effective_history_window,
                )
            else:
                tutor_prompt = generate_tutor_prompt(
                    conversation_history=conversation,
                    scenario_id=resolved_scenario_id,
                    template_path=resolved_tutor_prompt_path,
                    parameters=active_parameters,
                    definitions=definitions,
                    problem=problem_text,
                    topic=topic_text,
                    domain=domain_text,
                    difficulty=difficulty_text,
                    learning_objective=learning_obj,
                    target_misconception=target_miscon,
                    history_window=effective_history_window,
                )

            if mock:
                if is_finetuned:
                    if i == 0:
                        raw_tutor_reply = (
                            "<reasoning>Student made the common intuitive mistake of subtracting 1 from 1.10. Needs guidance to check total cost.</reasoning>\n"
                            "<state>Diagnosis</state>\n"
                            "<response>Let's check that. If the ball is 10 cents, and the bat costs $1.00 more than the ball, how much would the bat cost?</response>"
                        )
                    elif i == 1:
                        raw_tutor_reply = (
                            "<reasoning>Student self-corrected and found 5 cents. Now prompt them to articulate the general rule.</reasoning>\n"
                            "<state>Support</state>\n"
                            "<response>Spot on! In general, if you have two items where one costs $D more than the other and the total is $T, what is the general formula to find the smaller item's cost?</response>"
                        )
                    else:
                        raw_tutor_reply = (
                            "<reasoning>Student formulated the general rule correctly.</reasoning>\n"
                            "<state>Closing</state>\n"
                            "<response>Excellent deduction! You've mastered this concept completely. Ready for the next challenge?</response>"
                        )
                else:
                    if i == 0:
                        raw_tutor_reply = json.dumps({
                            "reasoning": "Student made the common intuitive mistake of subtracting 1 from 1.10. Needs guidance to check total cost.",
                            "end": 0,
                            "response": "Let's check that. If the ball is 10 cents, and the bat costs $1.00 more than the ball, how much would the bat cost?",
                        })
                    elif i == 1:
                        raw_tutor_reply = json.dumps({
                            "reasoning": "Student self-corrected and found 5 cents. Now prompt them to articulate the general rule.",
                            "end": 0,
                            "response": "Spot on! In general, if you have two items where one costs $D more than the other and the total is $T, what is the general formula to find the smaller item's cost?",
                        })
                    else:
                        raw_tutor_reply = json.dumps({
                            "reasoning": "Student formulated the general rule correctly.",
                            "end": 1,
                            "response": "Excellent deduction! You've mastered this concept completely. Ready for the next challenge?",
                        })
                tutor_tokens = 50
                success = True
                err = None
            else:
                success, raw_tutor_reply, tutor_tokens, err = tutor_model.call_model(tutor_prompt)

            if not success or not raw_tutor_reply:
                last_tutor_err = f"model call error: {err}"
                logger.warning(
                    "Tutor model [%s (%s)] call FAILED on turn %d (attempt %d/%d): error=%s",
                    tutor_model.model_name,
                    tutor_model.provider,
                    turn_tutor,
                    retry + 1,
                    max_tutor_retries,
                    err,
                )
                continue

            if is_finetuned:
                is_valid, parsed, val_err = validate_finetuned_tutor_turn(raw_tutor_reply, active_parameters)
            else:
                is_valid, parsed, val_err = validate_tutor_turn(raw_tutor_reply, active_parameters)

            if is_valid and parsed:
                tutor_turn_valid = True
                tutor_payload = parsed
                state_str = f", state: {parsed['state']}" if "state" in parsed else ""
                logger.info(
                    "Tutor model [%s (%s)] call SUCCESS on turn %d (end: %s%s, %d tokens):\nReasoning: %s\nResponse: \"%s\"",
                    tutor_model.model_name,
                    tutor_model.provider,
                    turn_tutor,
                    parsed["end"],
                    state_str,
                    tutor_tokens or 0,
                    parsed.get("reasoning", ""),
                    parsed.get("response", ""),
                )
                break
            else:
                last_tutor_err = val_err
                logger.warning(
                    "Tutor model [%s (%s)] response REJECTED on turn %d (attempt %d/%d): %s | Raw output: %s",
                    tutor_model.model_name,
                    tutor_model.provider,
                    turn_tutor,
                    retry + 1,
                    max_tutor_retries,
                    val_err,
                    raw_tutor_reply,
                )

        if not tutor_turn_valid or not tutor_payload:
            logger.error(
                "Tutor model [%s (%s)] failed after %d retries on turn %d. Stopping pipeline.",
                tutor_model.model_name,
                tutor_model.provider,
                max_tutor_retries,
                turn_tutor,
            )
            tutor_model.write_rejected_response(
                conversation,
                data_dir=data_dir,
                generator_id=generator_id,
                reason=f"Tutor failed structural validation on turn {turn_tutor}: {last_tutor_err}",
            )
            return False, None, {"error": f"tutor failed validation on turn {turn_tutor}: {last_tutor_err}"}

        end_signal = tutor_payload["end"]
        tutor_entry: Dict[str, Any] = {
            "turn": turn_tutor,
            "speaker": "tutor",
            "reasoning": tutor_payload["reasoning"],
            "end": end_signal,
            "response": tutor_payload["response"],
        }
        if "state" in tutor_payload:
            tutor_entry["state"] = tutor_payload["state"]
        if "raw_output" in tutor_payload:
            tutor_entry["raw_output"] = tutor_payload["raw_output"]

        conversation.append(tutor_entry)
        state_log = f", state: {tutor_payload.get('state')}" if "state" in tutor_payload else ""
        logger.info("Tutor turn %d recorded (end: %s%s)", turn_tutor, end_signal, state_log)

        # -------------------------------------------------------------
        # 3. Check for completion (end == 1)
        # -------------------------------------------------------------
        if end_signal == 1:
            closed = True
            logger.info("Conversation concluded naturally with 'end': 1 at round %d", rounds_completed)
            break

    if require_closing and not closed:
        logger.warning("Conversation did not reach 'end': 1 within %d iterations; rejecting.", max_iterations)
        tutor_model.write_rejected_response(
            conversation,
            data_dir=data_dir,
            generator_id=generator_id,
            reason="Did not reach end: 1 within max_iterations",
        )
        return False, None, {"error": "did not reach end: 1"}

    # -----------------------------------------------------------------
    # Log the complete conversation transcript with spacing
    # -----------------------------------------------------------------
    transcript_lines = [
        "",
        "=" * 70,
        "COMPLETE CONVERSATION TRANSCRIPT",
        f"  Tutor Model   : {tutor_model.provider} - {tutor_model.model_name}",
        f"  Tutor Prompt  : {tutor_prompt_label}",
        f"  Student Model : {student_model.provider} - {student_model.model_name} ({resolved_persona_id} - {persona_name})",
        f"  Scenario ID   : {resolved_scenario_id}",
        f"  Problem       : {problem_text}",
        f"  History Window: {effective_history_window if effective_history_window is not None else 'full'}",
        f"  Total Turns   : {len(conversation)} (Closed: {closed})",
        "=" * 70,
        "",
    ]
    for turn_item in conversation:
        turn_num = turn_item.get("turn", "?")
        spk = turn_item.get("speaker", "unknown").upper()
        if spk == "STUDENT":
            transcript_lines.append(f"Turn {turn_num} [STUDENT - {student_model.model_name}]:")
            transcript_lines.append(f"  {turn_item.get('text', '')}")
            transcript_lines.append("")
        else:
            end_val = turn_item.get("end", 0)
            state_val = turn_item.get("state")
            state_label = f", state={state_val}" if state_val else ""
            end_label = f" (end={end_val}{state_label})"
            transcript_lines.append(f"Turn {turn_num} [TUTOR{end_label} - {tutor_model.model_name}]:")
            if turn_item.get("reasoning"):
                transcript_lines.append(f"  Reasoning: {turn_item.get('reasoning')}")
            if state_val:
                transcript_lines.append(f"  State    : {state_val}")
            transcript_lines.append(f"  Response : {turn_item.get('response', turn_item.get('text', ''))}")
            transcript_lines.append("")

    transcript_lines.append("=" * 70)
    transcript_lines.append("")
    logger.info("\n".join(transcript_lines))

    # -----------------------------------------------------------------
    # Save conversation to data folder
    # -----------------------------------------------------------------
    config_dict = {
        "tutor_model": tutor_model.model_name,
        "tutor_provider": tutor_model.provider,
        "tutor_type": "finetuned_socratic" if is_finetuned else getattr(tutor_model, "tutor_type", None) or "base",
        "tutor_prompt": tutor_prompt_label,
        "history_window": effective_history_window,
        "student_model": student_model.model_name,
        "student_provider": student_model.provider,
        "max_iterations": max_iterations,
        "completed_rounds": rounds_completed,
        "closed": closed,
        "eval_mode": eval_mode,
        "parameters": active_parameters,
    }
    if eval_mode and eval_combo_data:
        config_dict["eval_combo_id"] = eval_combo_data.get("combo_id")
        config_dict["eval_combo_index"] = eval_combo_index

    record = {
        "config": config_dict,
        "scenario_id": resolved_scenario_id,
        "domain": domain_text,
        "topic": topic_text,
        "problem": problem_text,
        "conversation": conversation,
    }

    save_path = tutor_model.write_response(record, data_dir=data_dir, generator_id=generator_id)
    logger.info("Saved conversation to %s", save_path)
    return True, save_path, record
