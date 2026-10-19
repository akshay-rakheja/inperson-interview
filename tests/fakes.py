"""Injectable fake model clients. These test adapters and state handling only - never semantics."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from procurement.llm import LLMError


class FakeClient:
    provider = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    is_fake = True

    def __init__(self, responses: list[Any] | Callable[..., Any], model: str = "fake/model"):
        self.model = model
        self._responses = responses
        self.calls: list[dict] = []
        self.last_metadata: dict = {}

    def generate(self, *, task: str, payload: dict, schema: type[BaseModel]) -> BaseModel:
        self.calls.append({"task": task, "payload": payload, "schema": schema})
        self.last_metadata = {"task": task, "fake": True, "requested_model": self.model}
        if callable(self._responses):
            item = self._responses(task=task, payload=payload, schema=schema)
        else:
            item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return schema.model_validate(item)
        return item


def ref(case_id: str, path: str, sha: str, start: int, end: int, quote: str) -> dict:
    return {"source_id": f"{case_id}:{path}@{sha[:12]}", "start_line": start, "end_line": end, "quote": quote}


def src_ref(sources, path: str, start: int, end: int, quote: str) -> dict:
    src = next(s for s in sources if s.relative_path == path)
    return {"source_id": src.source_id, "start_line": start, "end_line": end, "quote": quote}


def assessment(case_id: str, qid: str, *, title: str = "Item", positions=None, gaps=None, next_action="",
               owner: str | None = "Anthony (Engineering)", deps=None, commitments=None,
               evidence_state: str = "partial", question_ids=None, origin: str = "questionnaire") -> dict:
    return {
        "requirement_key": f"{case_id}:{qid}",
        "question_ids": question_ids if question_ids is not None else [qid],
        "title": title,
        "workstream": "Security",
        "origin": origin,
        "current_position": positions or [],
        "evidence_state": evidence_state,
        "gaps": gaps or [],
        "dependencies": deps or [],
        "next_action": next_action,
        "suggested_owner": owner,
        "commitments": commitments or [],
    }


def position(text: str, refs: list[dict], kind: str = "reported", applicability: str = "customer_specific",
             date_text=None, effective_date_iso=None) -> dict:
    return {"text": text, "kind": kind, "applicability": applicability, "date_text": date_text,
            "effective_date_iso": effective_date_iso, "refs": refs}


def analysis(case_id: str, as_of: str, assessments: list[dict], unassessed=None, warnings=None) -> dict:
    return {"schema_version": "1", "case_id": case_id, "as_of_date": as_of, "summary": "Fake summary.",
            "assessments": assessments, "unassessed_question_ids": unassessed or [], "warnings": warnings or []}


class LLMErrors:
    auth = LLMError("auth", "OpenRouter rejected the key (401).")
    credits = LLMError("credits", "insufficient credits (402)")
    schema = LLMError("unsupported_schema", "structured output not supported")
    timeout = LLMError("timeout", "timed out")
