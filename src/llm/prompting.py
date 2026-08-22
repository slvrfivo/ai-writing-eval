"""Load and render the versioned official scoring prompt snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_VERSION = "writing_scoring_2026-07-20"
PROMPT_TEXT_PLACEHOLDER = "{{PROMPT_TEXT}}"
ESSAY_TEXT_PLACEHOLDER = "{{ESSAY_TEXT}}"


@dataclass(frozen=True)
class PromptSnapshot:
    version: str
    system: str
    user_template: str
    metadata: dict[str, Any]
    sha256: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def snapshot_directory(version: str = DEFAULT_PROMPT_VERSION) -> Path:
    if not version or Path(version).name != version:
        raise ValueError("prompt version must be a non-empty directory name")
    return project_root() / "prompts" / "official" / version


def calculate_snapshot_sha256(system: bytes, user: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(system)
    digest.update(b"\0")
    digest.update(user)
    return digest.hexdigest()


def _validate_template(user_template: str) -> None:
    counts = {
        PROMPT_TEXT_PLACEHOLDER: user_template.count(PROMPT_TEXT_PLACEHOLDER),
        ESSAY_TEXT_PLACEHOLDER: user_template.count(ESSAY_TEXT_PLACEHOLDER),
    }
    invalid = {placeholder: count for placeholder, count in counts.items() if count != 1}
    if invalid:
        raise ValueError(f"prompt placeholders must each occur exactly once: {invalid}")


def load_prompt_snapshot(
    version: str = DEFAULT_PROMPT_VERSION,
) -> PromptSnapshot:
    directory = snapshot_directory(version)
    system_bytes = (directory / "system.txt").read_bytes()
    user_bytes = (directory / "user.txt").read_bytes()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    system = system_bytes.decode("utf-8")
    user_template = user_bytes.decode("utf-8")
    _validate_template(user_template)

    if metadata.get("prompt_version") != version:
        raise ValueError("prompt version does not match metadata")

    sha256 = calculate_snapshot_sha256(system_bytes, user_bytes)
    if metadata.get("snapshot_sha256") != sha256:
        raise ValueError("prompt snapshot SHA-256 does not match metadata")

    return PromptSnapshot(
        version=version,
        system=system,
        user_template=user_template,
        metadata=metadata,
        sha256=sha256,
    )


def render_user_prompt(
    user_template: str,
    *,
    prompt_text: str,
    essay_text: str,
) -> str:
    if not isinstance(prompt_text, str) or not isinstance(essay_text, str):
        raise TypeError("prompt_text and essay_text must be strings")
    _validate_template(user_template)

    rendered = user_template.replace(PROMPT_TEXT_PLACEHOLDER, prompt_text)
    rendered = rendered.replace(ESSAY_TEXT_PLACEHOLDER, essay_text)
    if PROMPT_TEXT_PLACEHOLDER in rendered or ESSAY_TEXT_PLACEHOLDER in rendered:
        raise ValueError("an unresolved prompt placeholder remains after rendering")
    return rendered


def build_messages(
    prompt_text: str,
    essay_text: str,
    *,
    version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    snapshot = load_prompt_snapshot(version)
    return [
        {"role": "system", "content": snapshot.system},
        {
            "role": "user",
            "content": render_user_prompt(
                snapshot.user_template,
                prompt_text=prompt_text,
                essay_text=essay_text,
            ),
        },
    ]
