#!/usr/bin/env python3
"""Run conversation testing pipeline between a student model and tutor model."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    get_model_by_index_or_name,
    load_models_from_file,
    run_single_conversation,
)
from src.prompt_generator import (
    get_eval_combo,
    load_eval_combos,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-turn conversational evaluation between student and tutor models."
    )
    parser.add_argument(
        "--tutors-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "tutors.json"),
        help="Path to tutors JSON file.",
    )
    parser.add_argument(
        "--students-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "students.json"),
        help="Path to students JSON file.",
    )
    parser.add_argument(
        "--tutor-prompt",
        "--tutor-prompt-file",
        "--tutor-prompt-path",
        dest="tutor_prompt",
        type=str,
        default=None,
        help="Path or filename of tutor prompt markdown file (default: model's tutor_prompt or socratic_tutor_prompt_small.md).",
    )
    parser.add_argument(
        "--tutor-index",
        type=int,
        default=0,
        help="Index of tutor model in tutors.json (default: 0).",
    )
    parser.add_argument(
        "--student-index",
        type=int,
        default=0,
        help="Index of student model in students.json (default: 0).",
    )
    parser.add_argument(
        "--tutor-model",
        type=str,
        default=None,
        help="Optional name substring to select tutor model instead of index.",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default=None,
        help="Optional name substring to select student model instead of index.",
    )
    parser.add_argument(
        "--tutor-type",
        type=str,
        default=None,
        help="Optional tutor type override (e.g., 'finetuned_socratic' or 'base'). Defaults to model configuration in tutors.json.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=None,
        help="Sliding window size (number of recent turns) for tutor prompt. Defaults to model setting in tutors.json (or None for full history).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt override for fine-tuned tutor models ('default', 'pedagogical', or custom text).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of conversation sessions to generate (default: 1).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Max student-tutor rounds per conversation (default: 15).",
    )
    parser.add_argument(
        "--max-tutor-retries",
        type=int,
        default=5,
        help="Max retries for tutor to return valid structured JSON (default: 5).",
    )
    parser.add_argument(
        "--max-student-retries",
        type=int,
        default=5,
        help="Max retries for student model calls (default: 5).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Directory to store conversation JSON sets (default: data).",
    )
    parser.add_argument(
        "--generator-id",
        type=str,
        default=None,
        help="Generator ID for output file naming (defaults to DATASET_GENERATOR_ID env var).",
    )
    parser.add_argument(
        "--require-closing",
        action="store_true",
        help="Reject conversation if it hits max iterations without reaching Closing state.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model replies for testing without calling external APIs.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs/<provider>_<model_name>.log). Pass 'none' to disable file logging.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="random",
        help="Scenario ID from question_bank.json or 'random' (default: random).",
    )
    parser.add_argument(
        "--persona-id",
        type=str,
        default="random",
        help="Student persona ID (STU_001..STU_005) or 'random' (default: random).",
    )
    parser.add_argument(
        "--eval-mode",
        "-e",
        action="store_true",
        help="Enable evaluation mode: sequentially cycles through curated parameter combinations using modulo indexing for fair tutor model comparisons.",
    )
    parser.add_argument(
        "--eval-combos-path",
        type=str,
        default=None,
        help="Optional path to custom eval combos JSON file (defaults to eval_combos.json).",
    )

    args = parser.parse_args()

    # Load models
    tutors = load_models_from_file(args.tutors_path)
    students = load_models_from_file(args.students_path)

    tutor = get_model_by_index_or_name(
        tutors,
        index=None if args.tutor_model else args.tutor_index,
        name=args.tutor_model,
    )
    student = get_model_by_index_or_name(
        students,
        index=None if args.student_model else args.student_index,
        name=args.student_model,
    )

    # Resolve tutor prompt template path
    target_prompt = args.tutor_prompt or getattr(tutor, "tutor_prompt", None) or "socratic_tutor_prompt_small.md"
    tutor_prompt_path = Path(target_prompt)
    if not tutor_prompt_path.is_file():
        candidate = PROJECT_ROOT / target_prompt
        if candidate.is_file():
            tutor_prompt_path = candidate
        else:
            parser.error(f"Tutor prompt file not found: {target_prompt}")

    # Determine log file path (defaults to logs/{provider}_{model_name}.log based on tutor model)
    if args.log_file is None:
        log_file_path = str(PROJECT_ROOT / "logs" / f"{tutor.response_file_stem}.log")
    elif args.log_file.strip().lower() in ("none", "false", ""):
        log_file_path = None
    else:
        log_file_path = args.log_file

    # Configure logging handlers (console + optional log file)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    eval_combos = []
    if args.eval_mode:
        eval_combos = load_eval_combos(args.eval_combos_path)

    effective_window = args.history_window if args.history_window is not None else getattr(tutor, "history_window", None)

    print("\n" + "=" * 60)
    print("Conversation Testing Pipeline Configuration")
    print(f"  Tutor Model   : {tutor.provider} - {tutor.model_name}")
    print(f"  Tutor Prompt  : {tutor_prompt_path.name} ({tutor_prompt_path})")
    print(f"  History Window: {effective_window if effective_window is not None else 'full'}")
    print(f"  Student Model : {student.provider} - {student.model_name}")
    if args.eval_mode:
        print(f"  Mode          : EVAL MODE ({len(eval_combos)} sequential benchmark combos)")
    else:
        print(f"  Mode          : Standard Mode")
        print(f"  Student Persona: {args.persona_id if args.persona_id != 'random' else 'random (sampled per conversation)'}")
        print(f"  Scenario ID   : {args.scenario_id if args.scenario_id != 'random' else 'random (sampled per conversation)'}")
    print(f"  Sessions      : {args.iterations}")
    print(f"  Max Turns     : {args.max_iterations * 2} ({args.max_iterations} rounds)")
    print(f"  Data Output   : {args.data_dir}")
    print(f"  Log File      : {log_file_path if log_file_path else 'disabled'}")
    print(f"  Mock Mode     : {args.mock}")
    print("=" * 60 + "\n")

    successful = 0
    failed = 0
    failure_details = []
    saved_conversations = []

    for session_idx in range(args.iterations):
        convo_idx_1based = session_idx + 1
        if args.eval_mode:
            combo = get_eval_combo(convo_idx_1based, one_indexed=True, combos=eval_combos)
            combo_id = combo.get("combo_id", f"COMBO_{((convo_idx_1based - 1) % len(eval_combos)) + 1}")
            combo_num = ((convo_idx_1based - 1) % len(eval_combos)) + 1
            logger.info(
                "Starting session %d/%d [EVAL MODE: Combo %d/%d (%s)] -> Scenario: %s, Persona: %s",
                convo_idx_1based,
                args.iterations,
                combo_num,
                len(eval_combos),
                combo_id,
                combo.get("scenario_id"),
                combo.get("persona_id"),
            )
            print(
                f"\n>>> [Session {convo_idx_1based}/{args.iterations}] [EVAL MODE] "
                f"Combo {combo_num}/{len(eval_combos)} ({combo_id}): "
                f"Scenario={combo.get('scenario_id')}, Persona={combo.get('persona_id')} <<<"
            )
        else:
            logger.info("Starting session %d/%d", convo_idx_1based, args.iterations)

        success, save_path, record = run_single_conversation(
            tutor_model=tutor,
            student_model=student,
            max_iterations=args.max_iterations,
            max_tutor_retries=args.max_tutor_retries,
            max_student_retries=args.max_student_retries,
            data_dir=args.data_dir,
            generator_id=args.generator_id,
            require_closing=args.require_closing,
            mock=args.mock,
            log_file=log_file_path,
            scenario_id=args.scenario_id,
            persona_id=args.persona_id,
            tutor_prompt_path=tutor_prompt_path,
            eval_mode=args.eval_mode,
            eval_combo_index=convo_idx_1based,
            eval_combos=eval_combos if args.eval_mode else None,
            eval_combos_path=args.eval_combos_path,
            tutor_type=args.tutor_type,
            history_window=args.history_window,
            system_prompt=args.system_prompt,
        )
        if success and save_path:
            successful += 1
            saved_conversations.append(str(save_path))
            logger.info("[Session %d] Successfully saved to: %s", session_idx + 1, save_path)
            print(f"[Session {session_idx + 1}] Successfully saved to: {save_path}")
        else:
            failed += 1
            err_msg = record.get("error", "unknown error") if isinstance(record, dict) else "unknown error"
            failure_details.append({
                "session": session_idx + 1,
                "error": err_msg,
            })
            logger.warning("[Session %d] Failed: %s", session_idx + 1, err_msg)
            print(f"[Session {session_idx + 1}] Failed: {err_msg}")

    # -------------------------------------------------------------
    # Final Run Summary & Statistics (logged to file and console)
    # -------------------------------------------------------------
    total = args.iterations
    success_rate = (successful / total * 100.0) if total > 0 else 0.0

    summary_lines = [
        "=" * 60,
        "PIPELINE RUN SUMMARY",
        "=" * 60,
        f"  Tutor Model           : {tutor.provider} - {tutor.model_name}",
        f"  Student Model         : {student.provider} - {student.model_name}",
        f"  Tutor Prompt          : {tutor_prompt_path.name}",
        f"  Mode                  : {'EVAL MODE' if args.eval_mode else 'Standard Mode'}",
        f"  Total Sessions        : {total}",
        f"  Successfully Stored   : {successful}",
        f"  Failed Sessions       : {failed}",
        f"  Success Rate          : {success_rate:.1f}%",
    ]

    if saved_conversations:
        summary_lines.append("  Saved File Locations  :")
        unique_paths = sorted(set(saved_conversations))
        for p in unique_paths:
            count = saved_conversations.count(p)
            summary_lines.append(f"    - {p} ({count} conversation{'s' if count > 1 else ''})")

    if failure_details:
        summary_lines.append("  Failure Breakdown     :")
        for item in failure_details:
            summary_lines.append(f"    - Session {item['session']}: {item['error']}")

    summary_lines.append("=" * 60)
    summary_text = "\n" + "\n".join(summary_lines)

    logger.info(summary_text)
    print(f"\nSummary: {successful}/{total} conversations successfully saved ({failed} failed).\n")


if __name__ == "__main__":
    main()
