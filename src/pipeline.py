"""Conversation testing pipeline coordinating student and tutor models."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .model import Model
from .prompt_generator import generate_student_prompt, generate_tutor_prompt
from .validator import validate_student_turn, validate_tutor_turn

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
        for m in models:
            if target in m.model_name.lower():
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
) -> Tuple[bool, Optional[Path], Dict[str, Any]]:
    """
    Run a multi-turn conversation between a student model and a tutor model.

    Flow:
      For round 0 .. max_iterations-1:
        1. Student prompt generated -> Student model called (up to max_student_retries) -> validate response.
        2. Tutor prompt generated -> Tutor model called (up to max_tutor_retries) -> validate tutor JSON.
        3. If tutor state == "Closing", break immediately.
      Save valid conversation to data/{tutor_stem}/{generator_id}-set[n].json.
    """
    if log_file and log_file.strip().lower() not in ("none", "false", ""):
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

    problem_text = problem or "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    topic_text = topic or "Word problems & algebra"
    domain_text = domain or "Quantitative reasoning"

    logger.info(
        "Starting conversation test: Tutor=%s (%s), Student=%s (%s)",
        tutor_model.model_name,
        tutor_model.provider,
        student_model.model_name,
        student_model.provider,
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
            parameters=parameters,
            definitions=definitions,
            problem=problem_text,
            topic=topic_text,
            domain=domain_text,
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

            valid_student, student_text, student_err = validate_student_turn(raw_student_reply, parameters)
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

            tutor_prompt = generate_tutor_prompt(
                conversation_history=conversation,
                parameters=parameters,
                definitions=definitions,
                problem=problem_text,
                topic=topic_text,
                domain=domain_text,
            )

            if mock:
                if i == 0:
                    raw_tutor_reply = json.dumps({
                        "reasoning": "Student made the common intuitive mistake of subtracting 1 from 1.10. Needs guidance to check total cost.",
                        "state": "Support",
                        "response": "Let's check that. If the ball is 10 cents, and the bat costs $1.00 more than the ball, how much would the bat cost?",
                    })
                elif i == 1:
                    raw_tutor_reply = json.dumps({
                        "reasoning": "Student self-corrected and found 5 cents. Now prompt them to articulate the general rule.",
                        "state": "Synthesis",
                        "response": "Spot on! In general, if you have two items where one costs $D more than the other and the total is $T, what is the general formula to find the smaller item's cost?",
                    })
                else:
                    raw_tutor_reply = json.dumps({
                        "reasoning": "Student formulated the general rule correctly.",
                        "state": "Closing",
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

            is_valid, parsed, val_err = validate_tutor_turn(raw_tutor_reply, parameters)
            if is_valid and parsed:
                tutor_turn_valid = True
                tutor_payload = parsed
                logger.info(
                    "Tutor model [%s (%s)] call SUCCESS on turn %d (state: %s, %d tokens):\nReasoning: %s\nResponse: \"%s\"",
                    tutor_model.model_name,
                    tutor_model.provider,
                    turn_tutor,
                    parsed["state"],
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

        state = tutor_payload["state"]
        conversation.append({
            "turn": turn_tutor,
            "speaker": "tutor",
            "reasoning": tutor_payload["reasoning"],
            "state": state,
            "response": tutor_payload["response"],
        })
        logger.info("Tutor turn %d recorded (state: %s)", turn_tutor, state)

        # -------------------------------------------------------------
        # 3. Check for Closing state
        # -------------------------------------------------------------
        if state.lower() == "closing":
            closed = True
            logger.info("Conversation concluded naturally with 'Closing' state at round %d", rounds_completed)
            break

    if require_closing and not closed:
        logger.warning("Conversation did not reach 'Closing' state within %d iterations; rejecting.", max_iterations)
        tutor_model.write_rejected_response(
            conversation,
            data_dir=data_dir,
            generator_id=generator_id,
            reason="Did not reach Closing state within max_iterations",
        )
        return False, None, {"error": "did not reach Closing state"}

    # -----------------------------------------------------------------
    # Log the complete conversation transcript with spacing
    # -----------------------------------------------------------------
    transcript_lines = [
        "",
        "=" * 70,
        "COMPLETE CONVERSATION TRANSCRIPT",
        f"  Tutor Model   : {tutor_model.provider} - {tutor_model.model_name}",
        f"  Student Model : {student_model.provider} - {student_model.model_name}",
        f"  Problem       : {problem_text}",
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
            state_val = turn_item.get("state", "")
            state_label = f" ({state_val})" if state_val else ""
            transcript_lines.append(f"Turn {turn_num} [TUTOR{state_label} - {tutor_model.model_name}]:")
            if turn_item.get("reasoning"):
                transcript_lines.append(f"  Reasoning: {turn_item.get('reasoning')}")
            transcript_lines.append(f"  Response : {turn_item.get('response', turn_item.get('text', ''))}")
            transcript_lines.append("")

    transcript_lines.append("=" * 70)
    transcript_lines.append("")
    logger.info("\n".join(transcript_lines))

    # -----------------------------------------------------------------
    # Save conversation to data folder
    # -----------------------------------------------------------------
    record = {
        "config": {
            "tutor_model": tutor_model.model_name,
            "tutor_provider": tutor_model.provider,
            "student_model": student_model.model_name,
            "student_provider": student_model.provider,
            "max_iterations": max_iterations,
            "completed_rounds": rounds_completed,
            "closed": closed,
            "parameters": parameters or {},
        },
        "domain": domain_text,
        "topic": topic_text,
        "problem": problem_text,
        "conversation": conversation,
    }

    save_path = tutor_model.write_response(record, data_dir=data_dir, generator_id=generator_id)
    logger.info("Saved conversation to %s", save_path)
    return True, save_path, record
