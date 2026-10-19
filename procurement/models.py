"""Pydantic contracts shared across ingestion, analysis, drafting and updates.

Model-facing schemas (CaseAnalysis, CustomerUpdate) avoid free-form dicts so they can be
sent as strict JSON schemas. Dates are nullable strings: unknown dates stay null.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal["text", "markdown", "csv", "response"]


class SourceRecord(BaseModel):
    """One immutable source version. `text` is the canonical LF representation."""

    source_id: str
    case_id: str
    relative_path: str
    content_sha256: str
    text: str
    kind: SourceKind
    metadata: dict[str, Any] = Field(default_factory=dict)

    def lines(self) -> list[str]:
        return canonical_lines(self.text)


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    start_line: int = Field(description="1-based first line of the cited span")
    end_line: int = Field(description="1-based last line of the cited span (inclusive)")
    quote: str = Field(description="Verbatim text copied from within the cited lines")


def canonical_lines(text: str) -> list[str]:
    """Line anchors: split LF-normalised text; a trailing newline adds no empty line."""
    if text == "":
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


# ---------------------------------------------------------------------------
# Case analysis (issue #3)
# ---------------------------------------------------------------------------

PositionKind = Literal["proposed", "reported", "historical", "documented"]
Applicability = Literal["customer_specific", "generic", "other_customer", "unknown"]
EvidenceState = Literal["found", "partial", "missing", "conflicting"]
Origin = Literal["explicit_request", "questionnaire", "reference_suggestion", "clarification"]
DateKind = Literal["suggested", "reported_owner_commitment", "reported_customer_commitment", "unknown"]


class PositionStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="One narrowly-scoped factual statement with its qualifiers kept")
    kind: PositionKind
    applicability: Applicability
    date_text: str | None = Field(description="Date exactly as written in the source, or null")
    effective_date_iso: str | None = Field(description="YYYY-MM-DD only if the source states a full date, else null")
    refs: list[EvidenceRef]


class Dependency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    basis: Literal["explicit", "inferred"]
    refs: list[EvidenceRef]


class Commitment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    owner: str | None
    date_iso: str | None
    date_text: str | None
    date_kind: DateKind
    refs: list[EvidenceRef]


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_key: str = Field(description="<case_id>:Q-### for questionnaire-led items, or <case_id>:<source-anchor>")
    question_ids: list[str]
    title: str
    workstream: str
    origin: Origin
    current_position: list[PositionStatement]
    evidence_state: EvidenceState
    gaps: list[str]
    dependencies: list[Dependency]
    next_action: str
    suggested_owner: str | None
    commitments: list[Commitment]


class CaseAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"]
    case_id: str
    as_of_date: str
    summary: str
    assessments: list[Assessment]
    unassessed_question_ids: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Customer update drafting (issue #5)
# ---------------------------------------------------------------------------


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    refs: list[EvidenceRef]


class CustomerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    claims: list[DraftClaim]
    unresolved_items: list[str]
