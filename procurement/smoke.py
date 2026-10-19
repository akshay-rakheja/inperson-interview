"""Explicit, one-off authenticated schema check: `python -m procurement.smoke`.

Sends one small strict-schema request (nested refs, enums, nullable dates) through the same
adapter used for analysis. Prints safe metadata only - never the key.
"""

from __future__ import annotations

import json
import sys

from pydantic import BaseModel

from . import config
from .llm import LLMError, client_from_env
from .models import PositionStatement


class SmokeResult(BaseModel):
    positions: list[PositionStatement]
    note: str | None


def main() -> int:
    config.load_env()
    client = client_from_env()
    if client is None:
        print(f"UNRUN: no {config.KEY_VAR} configured in {config.ENV_PATH}")
        return 2
    source = "1| Dev: app logs 14d intended; traces configured separately.\n2| Certificate expires Oct 1, 2026."
    payload = {
        "system": "Extract narrowly-scoped position statements from the source. Cite source_id 'demo:src' with "
                  "verbatim quotes (without the 'n| ' prefix). Use null for unknown dates.",
        "user": f"<source id=\"demo:src\">\n{source}\n</source>",
        "max_tokens": 2000,
    }
    try:
        result = client.generate(task="schema_smoke", payload=payload, schema=SmokeResult)
    except LLMError as exc:
        print(json.dumps({"status": "failed", "kind": exc.kind, "error": exc.message[:500],
                          "metadata": client.last_metadata}, indent=2))
        return 1
    print(json.dumps({"status": "success", "metadata": client.last_metadata,
                      "result": result.model_dump()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
