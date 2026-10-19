"""Explicit local configuration: project-root .env loading with process-env precedence.

Only OPENROUTER_API_KEY and PROCUREMENT_MODEL are configurable. OPENAI_API_KEY is never
read. A blank or missing key is a visible state, not an error, and never triggers a request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "workspace.sqlite"

PROVIDER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-5.5"
KEY_VAR = "OPENROUTER_API_KEY"
MODEL_VAR = "PROCUREMENT_MODEL"

ENV_TEMPLATE = (
    "# Paste your OpenRouter API key after the equals sign. This file is ignored by Git.\n"
    f"{KEY_VAR}=\n"
    f"{MODEL_VAR}={DEFAULT_MODEL}\n"
)


def ensure_env_file(path: Path | None = None) -> bool:
    """Create the local .env template only if absent. Returns True if created.

    Never rewrites an existing file, so user-entered values are preserved.
    """
    path = path or ENV_PATH
    if path.exists():
        return False
    path.write_text(ENV_TEMPLATE, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return True


def load_env(path: Path | None = None) -> None:
    """Load the project-root .env explicitly; existing process variables win."""
    load_dotenv(dotenv_path=path or ENV_PATH, override=False)


@dataclass(frozen=True)
class Settings:
    has_key: bool
    model: str
    provider: str = PROVIDER
    base_url: str = OPENROUTER_BASE_URL
    env_path: str = ""

    @property
    def status_message(self) -> str:
        if self.has_key:
            return f"OpenRouter key loaded; model `{self.model}`."
        return (
            f"No OpenRouter key configured. Paste it after `{KEY_VAR}=` in `{self.env_path}` "
            "and restart the app. Analysis and drafting are disabled until then."
        )


def get_settings() -> Settings:
    key = os.environ.get(KEY_VAR, "").strip()
    model = os.environ.get(MODEL_VAR, "").strip() or DEFAULT_MODEL
    return Settings(has_key=bool(key), model=model, env_path=str(ENV_PATH))


def get_api_key() -> str:
    """Return the OpenRouter key or an empty string. Callers must never log it."""
    return os.environ.get(KEY_VAR, "").strip()
