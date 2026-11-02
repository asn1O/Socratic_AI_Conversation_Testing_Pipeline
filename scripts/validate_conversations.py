#!/usr/bin/env python3
"""Validate all conversation JSON files in the data directory."""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validator import is_valid_conversation_dataset

LOG_FORMAT = "%(asctime)s  %(levelname)s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("validator")

DATA_DIR = PROJECT_ROOT / "data"


def validate_file(file_path: Path, error_stats: Counter, delete_invalid: bool) -> tuple[int, int]:
    """Validate every conversation in a single JSON file. Returns (total_count, invalid_count)."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            conversations: list[dict] = json.load(f)
        except json.JSONDecodeError as exc:
            log.error("Failed to parse JSON file %s: %s", file_path, exc)
            return 0, 1

    if not isinstance(conversations, list):
        log.error("File %s does not contain a JSON array", file_path)
        return 0, 1

    valid = []
    invalid_count = 0
    total_count = len(conversations)

    for index, element in enumerate(conversations):
        is_valid, error_detail, error_category = is_valid_conversation_dataset(element)

        if is_valid:
            valid.append(element)
        else:
            invalid_count += 1
            error_stats[error_category] += 1
            action = "REMOVED" if delete_invalid else "INVALID"
            log.warning("%s  file=%s  index=%d  reason=%s", action, file_path, index, error_detail)

    if delete_invalid and invalid_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(valid, f, indent=2, ensure_ascii=False)
            f.write("\n")
        log.info("Rewrote %s (removed %d elements)", file_path, invalid_count)

    return total_count, invalid_count


def validate_dataset(
    data_dir: Path = DATA_DIR,
    delete_invalid: bool = False,
) -> tuple[int, int, dict[str, dict[str, int]], Counter, int]:
    """Validate all conversation JSON files in data_dir."""
    json_files = sorted([
        f for f in data_dir.rglob("*.json")
        if not f.name.endswith("flagged_config_leakage.json")
    ])
    log.info("Found %d JSON files under %s", len(json_files), data_dir)

    total_conversations = 0
    total_invalid = 0
    error_stats = Counter()
    folder_stats: dict[str, dict[str, int]] = {}

    if data_dir.exists():
        for sub_dir in sorted(data_dir.iterdir()):
            if sub_dir.is_dir() and not sub_dir.name.startswith("."):
                folder_stats[sub_dir.name] = {"total": 0, "invalid": 0}

    for file_path in json_files:
        try:
            rel_path = file_path.relative_to(data_dir)
            folder_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "(root)"
        except ValueError:
            folder_name = file_path.parent.name

        if folder_name not in folder_stats:
            folder_stats[folder_name] = {"total": 0, "invalid": 0}

        file_total, file_invalid = validate_file(file_path, error_stats, delete_invalid)
        total_conversations += file_total
        total_invalid += file_invalid
        folder_stats[folder_name]["total"] += file_total
        folder_stats[folder_name]["invalid"] += file_invalid

    return total_conversations, total_invalid, folder_stats, error_stats, len(json_files)


def main():
    parser = argparse.ArgumentParser(description="Validate stored conversation JSON files in data folder.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to the directory containing conversation JSON files (default: data).",
    )
    parser.add_argument(
        "--delete-invalid",
        action="store_true",
        help="Remove invalid elements from files.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    total_convs, total_inv, folder_stats, error_stats, file_count = validate_dataset(
        data_dir, args.delete_invalid
    )

    log.info("Validation complete. Total files: %d, Conversations: %d, Invalid: %d", file_count, total_convs, total_inv)

    if folder_stats:
        log.info("── Folder Summary ──")
        max_name_len = max(len(name) for name in folder_stats)
        for folder_name, stats in sorted(folder_stats.items()):
            inv = stats["invalid"]
            tot = stats["total"]
            val = tot - inv
            pct = (inv / tot * 100) if tot > 0 else 0.0
            log.info(
                "  %-*s : %4d invalid / %4d total (%4d valid | %5.1f%% invalid)",
                max_name_len,
                folder_name,
                inv,
                tot,
                val,
                pct,
            )

    if error_stats:
        log.info("── Error Categories ──")
        for error_msg, count in error_stats.most_common():
            log.info("  %4d × %s", count, error_msg)
    elif total_convs > 0:
        log.info("All inspected conversations are 100%% structurally valid!")


if __name__ == "__main__":
    main()
