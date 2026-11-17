# Conversation Testing Pipeline

A dedicated testing framework to evaluate how effectively tutor models conduct Socratic conversations when interacting with an LLM impersonating a student. The pipeline executes multi-turn conversations, enforces structured tutor turn validation with automatic retries, monitors for prompt config leakage, and saves completed conversations in set-based JSON files partitioned by tutor model.

---

## Directory Structure

```
Conversation_testing_pipeline/
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── tutor_prompt.md           # Tutor instructions template (empty for now)
├── student_prompt.md         # Student instructions template (empty for now)
├── models/
│   ├── tutors.json           # Available tutor models (Google, NVIDIA, OpenRouter)
│   └── students.json         # Available student impersonation models
├── src/
│   ├── __init__.py
│   ├── model.py              # LLM client, rate limiting, and JSON set storage
│   ├── prompt_generator.py   # Extensible student & tutor prompt builders
│   ├── validator.py          # Turn-level validation & config leakage detection
│   └── pipeline.py           # Multi-turn conversation loop coordinator
├── scripts/
│   ├── run_pipeline.py       # CLI runner supporting --tutor-index, --student-index
│   └── validate_conversations.py # Standalone dataset auditor for data/
├── data/                     # Stored conversations partitioned by tutor model
└── tests/
    ├── test_validator.py     # Unit tests for turn validations & leakage checks
    ├── test_prompt_generator.py # Unit tests for prompt generation & injection
    └── test_pipeline.py      # End-to-end tests for the conversation loop
```

---

## How It Works

### 1. Multi-Turn Interaction Loop
For each conversation (up to `max_iterations`, default: 15 rounds / 30 turns):
1. **Student Turn**:
   - The student model is queried with the conversation history and task context.
   - The response is validated and recorded (`speaker: "student"`).
2. **Tutor Turn**:
   - The tutor model is queried with the updated conversation history.
   - The tutor must return a valid structured JSON response:
     ```json
     {
       "reasoning": "<rationale for pedagogical decision>",
       "end": 0,
       "response": "<message shown to student>"
     }
     ```
   - If the response is malformed, missing required fields, has an invalid `end` value (must be `0` or `1`), or leaks prompt configurations in reasoning, the tutor model is automatically retried (up to `max_tutor_retries`, default: 5).
3. **Closing Detection**:
   - When the tutor sets `"end": 1`, the loop concludes immediately.

### 2. Conversation Storage & Format
Valid conversations are stored under `data/{tutor_model_stem}/` in the format:
```
data/{provider}_{model_name}/{DATASET_GENERATOR_ID}-set[{set_index}].json
```
- Each set file contains a JSON array with up to **15 conversations**.
- When a file reaches 15 conversations, the pipeline automatically advances to the next set file (`set[1].json`, `set[2].json`, etc.).
- Rejected conversations or malformed turns are recorded in `data/rejected_responses/` for debugging.

---

## Evaluation Mode (Fair Benchmarking Across Tutor Models)

To compare different tutor models fairly, run with `--eval-mode` (or `-e`). Instead of sampling questions and student personas randomly, Evaluation Mode cycles through a deterministic list of curated parameter combinations (`eval_combos.json`) using sequential modulo indexing:

$$\text{combo\_index} = (i - 1) \pmod L$$

where $i \ge 1$ is the conversation session number and $L$ is the number of combos (10 by default).

- Session 1 $\to$ Combo 1 (`EVAL_01`): Physics (`PHY_001`) + Confident but Incorrect (`STU_002`)
- Session 2 $\to$ Combo 2 (`EVAL_02`): Math (`MAT_001`) + Procedural Learner (`STU_004`)
- ...
- Session 10 $\to$ Combo 10 (`EVAL_10`): Logic (`LOG_002`) + Passive Learner (`STU_003`)
- Session 11 $\to$ Combo 1 (`EVAL_01`): wraps around to 1st combo
- Session 12 $\to$ Combo 2 (`EVAL_02`): wraps around to 2nd combo

This guarantees that each tutor model interacts with a student LLM using the **identical parameters, questions, and misconceptions in the exact same order**.

---

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys and DATASET_GENERATOR_ID in .env
```

### 2. Run Pipeline

Run evaluation mode for 10 benchmark conversations:
```bash
python3 scripts/run_pipeline.py --tutor-index 0 --student-index 0 --eval-mode --iterations 10
```

Run with custom tutor prompt (e.g. `base_tutor_prompt.md`):
```bash
python3 scripts/run_pipeline.py --tutor-prompt base_tutor_prompt.md --eval-mode --iterations 10
```

Run standard mode with specific or random scenarios:
```bash
python3 scripts/run_pipeline.py --tutor-index 0 --student-index 0 --scenario-id PHY_001 --persona-id STU_002
```

Offline Mock Mode (test without consuming API credits):
```bash
python3 scripts/run_pipeline.py --mock --eval-mode --iterations 12
```

### 3. Audit Dataset
Run the standalone dataset validator over all saved conversations:
```bash
python3 scripts/validate_conversations.py
```
To remove any invalid conversations:
```bash
python3 scripts/validate_conversations.py --delete-invalid
```

### 4. Run Tests
```bash
pytest tests/ -v
```
