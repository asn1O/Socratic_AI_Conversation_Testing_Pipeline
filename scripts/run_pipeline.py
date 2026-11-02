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

    args = parser.parse_args()

    # Load models
    tutors = load_models_from_file(args.tutors_path)
    students = load_models_from_file(args.students_path)

    tutor = get_model_by_index_or_name(tutors, index=args.tutor_index, name=args.tutor_model)
    student = get_model_by_index_or_name(students, index=args.student_index, name=args.student_model)

    print("\n" + "=" * 60)
    print("Conversation Testing Pipeline Configuration")
    print(f"  Tutor Model   : {tutor.provider} - {tutor.model_name}")
    print(f"  Student Model : {student.provider} - {student.model_name}")
    print(f"  Sessions      : {args.iterations}")
    print(f"  Max Turns     : {args.max_iterations * 2} ({args.max_iterations} rounds)")
    print(f"  Data Output   : {args.data_dir}")
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
            data_dir=args.data_dir,
            generator_id=args.generator_id,
            require_closing=args.require_closing,
            mock=args.mock,
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
