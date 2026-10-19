"""Explicit command-line analysis run:
`python -m procurement.cli analyze <case_id> [YYYY-MM-DD] [--force] [--db PATH]`.

Uses the same store/adapter as the UI. Prints safe run metadata, never the key.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from . import config
from .analysis import run_case_analysis
from .ingest import discover_cases, import_case
from .llm import client_from_env
from .store import Store


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "analyze":
        print(__doc__)
        return 2
    config.load_env()
    client = client_from_env()
    if client is None:
        print(config.get_settings().status_message)
        return 2
    case_id = argv[1]
    as_of = date.fromisoformat(argv[2]) if len(argv) > 2 and not argv[2].startswith("--") else date.today()
    db = Path(argv[argv.index("--db") + 1]) if "--db" in argv else config.DEFAULT_DB_PATH
    store = Store(db)
    known = {c["case_id"] for c in discover_cases(config.REPO_ROOT)}
    if case_id not in known:
        print(f"Unknown case {case_id}; known: {sorted(known)}")
        return 2
    store.save_sources(case_id, import_case(config.REPO_ROOT, case_id))
    result = run_case_analysis(store=store, case_id=case_id, as_of_date=as_of, client=client,
                               force="--force" in argv)
    run = result["run"]
    meta = run.get("request_metadata", {})
    print(json.dumps({
        "run_id": run["run_id"], "status": run["status"], "reused": result["reused"], "error": run.get("error"),
        "model": run["model"], "returned_model": meta.get("returned_model"), "latency_s": meta.get("latency_s"),
        "finish_reason": meta.get("finish_reason"),
        "tokens": {k: (meta.get("usage") or {}).get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
        "assessments": len((run.get("analysis") or {}).get("assessments", [])),
        "analysis_mode": run.get("analysis_mode"), "mode_reason": run.get("analysis_mode_reason"),
        "reanalysed_keys": run.get("reanalysed_keys"), "carried": len(run.get("carried_keys") or []),
        "warnings": (run.get("analysis") or {}).get("warnings", []),
    }, indent=2))
    return 0 if run["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
