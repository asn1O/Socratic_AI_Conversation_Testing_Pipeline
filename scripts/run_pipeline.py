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
        default="socratic_tutor_prompt_small.md",
        help="Path or filename of tutor prompt markdown file (default: socratic_tutor_prompt_small.md).",
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
        default=str(PROJECT_ROOT / "logs" / "pipeline.log"),
        help="Path to log file (default: logs/pipeline.log). Pass 'none' to disable file logging.",
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

    args = parser.parse_args()

    # Configure logging handlers (console + optional log file)
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file and args.log_file.strip().lower() not in ("none", "false", ""):
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )

    # Load models
    tutors = load_models_from_file(args.tutors_path)
    students = load_models_from_file(args.students_path)

    # Resolve tutor prompt template path
    tutor_prompt_path = Path(args.tutor_prompt)
    if not tutor_prompt_path.is_file():
        candidate = PROJECT_ROOT / args.tutor_prompt
        if candidate.is_file():
            tutor_prompt_path = candidate
        else:
            parser.error(f"Tutor prompt file not found: {args.tutor_prompt}")

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

    print("\n" + "=" * 60)
    print("Conversation Testing Pipeline Configuration")
    print(f"  Tutor Model   : {tutor.provider} - {tutor.model_name}")
    print(f"  Tutor Prompt  : {tutor_prompt_path.name} ({tutor_prompt_path})")
    print(f"  Student Model : {student.provider} - {student.model_name}")
    print(f"  Student Persona: {args.persona_id if args.persona_id != 'random' else 'random (sampled per conversation)'}")
    print(f"  Scenario ID   : {args.scenario_id if args.scenario_id != 'random' else 'random (sampled per conversation)'}")
    print(f"  Sessions      : {args.iterations}")
    print(f"  Max Turns     : {args.max_iterations * 2} ({args.max_iterations} rounds)")
    print(f"  Data Output   : {args.data_dir}")
    print(f"  Log File      : {args.log_file if args.log_file else 'disabled'}")
    print(f"  Mock Mode     : {args.mock}")
    print("=" * 60 + "\n")

    successful = 0
    for session_idx in range(args.iterations):
        logger.info("Starting session %d/%d", session_idx + 1, args.iterations)
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
            log_file=args.log_file,
            scenario_id=args.scenario_id,
            persona_id=args.persona_id,
            tutor_prompt_path=tutor_prompt_path,
        )
        if success and save_path:
            successful += 1
            print(f"[Session {session_idx + 1}] Successfully saved to: {save_path}")
        else:
            print(f"[Session {session_idx + 1}] Failed: {record.get('error', 'unknown error')}")

    print("\n" + "=" * 60)
    print(f"Summary: {successful}/{args.iterations} conversations successfully saved.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
