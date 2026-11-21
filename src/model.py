from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    APIStatusError,
)

REQUEST_TIMEOUT_SECONDS = 120


@dataclass
class Model:
    provider: str
    api_key: str
    model_name: str
    url: str
    wait_1_min: bool = False
    last_use: Optional[datetime] = None
    expired: bool = False
    no_of_replies: int = 0
    avg_tokens_used: float = 0.0
    api_key_env: Optional[str] = None
    consecutive_failures: int = 0
    tutor_type: Optional[str] = None
    prompt_format: Optional[str] = None
    history_window: Optional[int] = None
    tutor_prompt: Optional[str] = None
    system_prompt: Optional[str] = None

    @property
    def is_finetuned_socratic(self) -> bool:
        """Check if model is configured as a fine-tuned Socratic tutor."""
        return (
            getattr(self, "tutor_type", None) == "finetuned_socratic"
            or getattr(self, "prompt_format", None) == "sft_xml"
        )

    def mark_expired(self, now: Optional[datetime] = None) -> None:
        """Mark the model as unusable."""
        self.expired = True
        if now is None:
            now = datetime.now(timezone.utc)
        self.last_use = now

    def increment_consecutive_failures(self) -> None:
        """Increment consecutive failure counter."""
        self.consecutive_failures += 1

    def reset_consecutive_failures(self) -> None:
        """Reset consecutive failure counter to 0."""
        self.consecutive_failures = 0

    def to_dict(self) -> dict:
        """Serialize Model to a JSON-serializable dictionary."""
        d = asdict(self)
        d["api_key"] = self.api_key_env or d["api_key"]
        d.pop("api_key_env", None)
        d["last_use"] = self.last_use.isoformat() if self.last_use else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "Model":
        """Create a Model from a dictionary, resolving the API key from environment."""
        last = d.get("last_use")
        last_use = datetime.fromisoformat(last) if last else None
        if last_use is not None and last_use.tzinfo is None:
            last_use = last_use.replace(tzinfo=timezone.utc)

        expired = bool(d.get("expired", False))
        api_key_env = d.get("api_key", "")
        # Resolve from environment variable if present, else use as-is
        resolved_key = os.getenv(api_key_env, api_key_env)
        history_window_val = d.get("history_window")
        history_window = int(history_window_val) if history_window_val is not None else None
        return Model(
            provider=d.get("provider", "Unknown"),
            api_key=resolved_key,
            model_name=d["model_name"],
            url=d.get("url", ""),
            wait_1_min=bool(d.get("wait_1_min", False)),
            last_use=last_use,
            expired=expired,
            no_of_replies=int(d.get("no_of_replies", 0)),
            avg_tokens_used=float(d.get("avg_tokens_used", 0.0) or 0.0),
            api_key_env=api_key_env,
            consecutive_failures=int(d.get("consecutive_failures", 0)),
            tutor_type=d.get("tutor_type"),
            prompt_format=d.get("prompt_format"),
            history_window=history_window,
            tutor_prompt=d.get("tutor_prompt"),
            system_prompt=d.get("system_prompt"),
        )

    def record_response(self, total_tokens: Optional[int] = None) -> None:
        """Update response counters and running average token usage."""
        if total_tokens is None:
            total_tokens = 0

        previous_replies = self.no_of_replies
        self.no_of_replies = previous_replies + 1
        self.avg_tokens_used = (
            ((self.avg_tokens_used * previous_replies) + int(total_tokens))
            / self.no_of_replies
            if self.no_of_replies > 0
            else 0.0
        )

    @property
    def response_file_stem(self) -> str:
        """Generate sanitized directory name for the model (e.g. Google_gemini-3.5-flash)."""
        return f"{self.provider}_{self.model_name}".replace("/", "_").replace(":", "_")

    def get_dest_write_path(self, data_dir: str = "data") -> Path:
        """Return the directory path for storing conversations generated with this tutor model."""
        return Path(data_dir) / self.response_file_stem

    def write_response(
        self,
        response: dict,
        data_dir: str = "data",
        generator_id: Optional[str] = None,
    ) -> Path:
        """Append a conversation response to a JSON set file containing at most 15 items."""
        response_dir = self.get_dest_write_path(data_dir)
        response_dir.mkdir(parents=True, exist_ok=True)

        contributor_id = (generator_id or os.getenv("DATASET_GENERATOR_ID", "test_runner")).strip()
        if not contributor_id or not re.fullmatch(r"[A-Za-z0-9_-]+", contributor_id):
            contributor_id = "test_runner"

        set_files = []
        for path in response_dir.glob(f"{contributor_id}-set[[]*.json"):
            match = re.fullmatch(
                rf"{re.escape(contributor_id)}-set\[(\d+)\]\.json",
                path.name,
            )
            if match:
                set_files.append((int(match.group(1)), path))

        if set_files:
            set_index, response_path = max(set_files, key=lambda item: item[0])
            with response_path.open("r", encoding="utf-8") as file:
                responses = json.load(file)
            if not isinstance(responses, list):
                responses = []
        else:
            set_index = 0
            response_path = response_dir / f"{contributor_id}-set[{set_index}].json"
            responses = []

        if len(responses) >= 15:
            set_index += 1
            response_path = response_dir / f"{contributor_id}-set[{set_index}].json"
            if response_path.exists():
                with response_path.open("r", encoding="utf-8") as file:
                    responses = json.load(file)
                if not isinstance(responses, list):
                    responses = []
            else:
                responses = []

        responses.append(response)
        with response_path.open("w", encoding="utf-8") as file:
            json.dump(responses, file, indent=2, ensure_ascii=False)
            file.write("\n")

        return response_path

    def write_rejected_response(
        self,
        response: object,
        data_dir: str = "data",
        generator_id: Optional[str] = None,
        reason: str = "unspecified",
    ) -> Path:
        """Append a rejected raw response or failed record to a text audit file."""
        rejected_dir = Path(data_dir) / "rejected_responses"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        contributor_id = (generator_id or os.getenv("DATASET_GENERATOR_ID", "test_runner")).strip()
        if not contributor_id or not re.fullmatch(r"[A-Za-z0-9_-]+", contributor_id):
            contributor_id = "test_runner"

        response_path = rejected_dir / f"{contributor_id}-{self.response_file_stem}.txt"
        response_text = (
            response
            if isinstance(response, str)
            else json.dumps(response, ensure_ascii=False, indent=2, default=str)
        )

        with response_path.open("a", encoding="utf-8") as file:
            if response_path.stat().st_size > 0:
                file.write("\n" + "=" * 60 + "\n")
            timestamp = datetime.now(timezone.utc).isoformat()
            file.write(f"[{timestamp}] REJECTED REASON: {reason}\n")
            file.write(response_text)
            file.write("\n")

        return response_path

    def call_model(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ) -> Tuple[bool, Optional[str], int, Optional[str]]:
        """
        Send a prompt or multi-turn messages array to the model and return:
            (success, reply, total_tokens, error_code)
        """
        try:
            parsed = urlparse(self.url or "")
            provider_host = parsed.netloc or self.url
        except Exception:
            provider_host = self.url

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.url if self.url else None,
                timeout=timeout,
            )

            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            self.last_use = datetime.now(timezone.utc)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            reply = response.choices[0].message.content if response.choices else None
            total_tokens = response.usage.total_tokens if response.usage else 0

            if not reply:
                return False, None, total_tokens or 0, "NO_RESPONSE"

            self.record_response(total_tokens)
            self.reset_consecutive_failures()
            return True, reply.strip(), total_tokens or 0, None

        except RateLimitError:
            self.last_use = datetime.now(timezone.utc)
            self.increment_consecutive_failures()
            return False, None, 0, "HTTP_429"

        except AuthenticationError:
            self.increment_consecutive_failures()
            return False, None, 0, "HTTP_401"

        except APIStatusError as exc:
            self.increment_consecutive_failures()
            return False, None, 0, f"HTTP_{exc.status_code}"

        except (APIConnectionError, APITimeoutError) as exc:
            self.increment_consecutive_failures()
            return False, None, 0, "TIMEOUT_OR_CONNECTION_ERROR"

        except Exception as exc:
            self.increment_consecutive_failures()
            return False, None, 0, f"EXCEPTION_{type(exc).__name__}"
