"""New information: immutable response sources, conservative draft invalidation, run comparison."""

from __future__ import annotations

import hashlib
import re

from .ingest import make_source_id
from .models import CaseAnalysis, SourceRecord
from .store import Store, now_iso


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "response"


def append_response(*, case_id: str, text: str, label: str, reported_author: str | None, simulated: bool,
                    related_keys: list[str], store: Store, effective_date_text: str | None = None) -> SourceRecord:
    """Persist a pasted response as a new immutable source, then mark existing drafts stale.

    Identical (label, author, simulated flag, body) resubmissions resolve to the same source_id,
    so a retry never duplicates the source.
    """
    body = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        raise ValueError("Response text is empty")
    if not label.strip():
        raise ValueError("A label is required")
    foreign = [k for k in related_keys if not k.startswith(f"{case_id}:")]
    if foreign:
        raise ValueError(f"Related keys belong to another case: {foreign}")
    header = (f"[{'SIMULATED DEMO INPUT' if simulated else 'PASTED RESPONSE'}] {label.strip()}"
              f" | reported author (unverified): {(reported_author or 'unknown').strip()}")
    full = f"{header}\n{body}\n"
    raw = full.encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    rel = f"responses/{sha[:12]}-{_slug(label)}.txt"
    source_id = make_source_id(case_id, rel, sha)

    existing = {s.source_id: s for s in store.get_source_versions(case_id)}
    if source_id in existing:
        return existing[source_id]
    record = SourceRecord(
        source_id=source_id, case_id=case_id, relative_path=rel, content_sha256=sha, text=full, kind="response",
        metadata={"origin": "pasted_response", "label": label.strip(), "reported_author": reported_author,
                  "author_verified": False, "simulated": simulated, "related_keys": related_keys,
                  "received_at": now_iso(), "effective_date_text": effective_date_text or None,
                  "line_count": full.count("\n"), "trailing_newline": True},
    )
    store.save_sources(case_id, [record])  # persisted before any model call
    store.invalidate_drafts(case_id, f"New information added: {label.strip()} ({source_id})")
    return record


def _positions(a) -> list[str]:
    return [f"{p.kind}/{p.applicability}: {p.text}" for p in a.current_position]


def _commitments(a) -> list[str]:
    return [f"{c.text} [{c.date_kind}; {c.date_iso or c.date_text or 'date unconfirmed'}]" for c in a.commitments]


def _dependencies(a) -> list[str]:
    return [f"{d.basis}: {d.description}" for d in a.dependencies]


FIELDS = {
    "evidence_state": lambda a: a.evidence_state,
    "current_position": _positions,
    "gaps": lambda a: list(a.gaps),
    "next_action": lambda a: a.next_action,
    "commitments": _commitments,
    "dependencies": _dependencies,
    "suggested_owner": lambda a: a.suggested_owner,
    "title": lambda a: a.title,
    "question_ids": lambda a: sorted(a.question_ids),
}


def diff_analyses(before: CaseAnalysis, after: CaseAnalysis) -> dict:
    """Compare by stable requirement key. Rewording shows as a change; this is not a semantic diff."""
    b = {a.requirement_key: a for a in before.assessments}
    a_ = {a.requirement_key: a for a in after.assessments}
    changed: dict[str, dict] = {}
    unchanged_open: list[str] = []
    for key in sorted(set(b) & set(a_)):
        delta = {}
        for name, get in FIELDS.items():
            old, new = get(b[key]), get(a_[key])
            if old != new:
                if isinstance(old, list):
                    delta[name] = {"before": old, "after": new,
                                   "added": [x for x in new if x not in old],
                                   "removed": [x for x in old if x not in new]}
                else:
                    delta[name] = {"before": old, "after": new}
        if delta:
            changed[key] = delta
        elif a_[key].gaps or a_[key].evidence_state != "found":
            unchanged_open.append(key)
    return {
        "added": sorted(set(a_) - set(b)),
        "removed": sorted(set(b) - set(a_)),
        "changed": changed,
        "unchanged_open": unchanged_open,
        "still_open_after": sorted(k for k, a in a_.items() if a.gaps or a.evidence_state != "found"),
        "unassessed_before": before.unassessed_question_ids,
        "unassessed_after": after.unassessed_question_ids,
    }


def mark_reviews_for_recheck(store: Store, case_id: str, new_run_id: str, analysis: CaseAnalysis) -> int:
    """After a successful reanalysis, every factual review note needs recheck; owner overrides persist."""
    keys = {a.requirement_key for a in analysis.assessments}
    count = 0
    for review in store.get_reviews(case_id):
        if not review.get("review_note") or review.get("recheck_against_run") == new_run_id:
            continue
        reason = ("requirement no longer present in the new run" if review["requirement_key"] not in keys
                  else "analysis refreshed; prior review applied to older text")
        store.save_review(case_id, review["requirement_key"],
                          {"needs_recheck": True, "recheck_against_run": new_run_id, "recheck_reason": reason})
        count += 1
    return count
