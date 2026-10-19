"""Deterministic "Actions by suggested owner" grouping. Zero model calls.

Groups existing assessment asks, gaps, refs, commitments and dependencies by owner. Ownership is
a suggestion or a local coordination choice - never an accepted assignment.
"""

from __future__ import annotations

from .models import CaseAnalysis

UNASSIGNED = "Unassigned"


def _resolve_owner(requirement_key: str, suggested: str | None, overrides: dict[str, str]) -> tuple[str, str]:
    override = (overrides.get(requirement_key) or "").strip()
    if override:
        return override, "local_override"
    if suggested and suggested.strip():
        return suggested.strip(), "suggested"
    return UNASSIGNED, "unassigned"


def build_owner_action_packs(*, analysis: CaseAnalysis, owner_overrides: dict[str, str] | None = None) -> list[dict]:
    prefix = f"{analysis.case_id}:"
    current_keys = {a.requirement_key for a in analysis.assessments}
    # Overrides for another case or a removed requirement are ignored.
    overrides = {k: v for k, v in (owner_overrides or {}).items() if k.startswith(prefix) and k in current_keys}

    groups: dict[str, list[dict]] = {}
    for a in analysis.assessments:
        ask = a.next_action.strip()
        if not a.gaps and not ask:
            continue
        if not ask:
            ask = f"Clarify missing information for {a.title}"
        owner, owner_basis = _resolve_owner(a.requirement_key, a.suggested_owner, overrides)

        refs, seen = [], set()
        for holder in [*a.current_position, *a.dependencies]:
            for r in holder.refs:
                ident = (r.source_id, r.start_line, r.end_line, r.quote)
                if ident not in seen:
                    seen.add(ident)
                    refs.append(r.model_dump())

        groups.setdefault(owner, []).append({
            "requirement_key": a.requirement_key,
            "title": a.title,
            "workstream": a.workstream,
            "evidence_state": a.evidence_state,
            "ask": ask,
            "gaps": list(a.gaps),
            "refs": refs,
            "commitments": [c.model_dump() for c in a.commitments],
            "dependencies": [d.model_dump() for d in a.dependencies],
            "owner_basis": owner_basis,
            "suggested_owner": a.suggested_owner,
        })

    ordered = sorted(groups.items(), key=lambda kv: (kv[0] == UNASSIGNED, kv[0].lower()))
    return [{"owner_label": owner, "items": sorted(items, key=lambda i: i["requirement_key"])}
            for owner, items in ordered]


def pack_as_text(pack: dict, *, case_id: str, run_id: str, outdated: bool) -> str:
    """Plain, copyable request text for one owner. No delivery happens."""
    lines = [f"Requests for {pack['owner_label']} - case {case_id} - basis run {run_id}"
             + (" (OUTDATED: based on an older run)" if outdated else ""), ""]
    for i, item in enumerate(pack["items"], start=1):
        lines.append(f"{i}. [{item['requirement_key']}] {item['title']}")
        lines.append(f"   Ask: {item['ask']}")
        for gap in item["gaps"]:
            lines.append(f"   Why / gap: {gap}")
        for dep in item["dependencies"]:
            lines.append(f"   Dependency ({dep['basis']}): {dep['description']}")
        for c in item["commitments"]:
            when = c.get("date_iso") or c.get("date_text") or "date unconfirmed"
            lines.append(f"   Source-reported commitment ({c['date_kind']}, {when}): {c['text']}")
        for r in item["refs"]:
            lines.append(f"   Source: {r['source_id']} L{r['start_line']}-{r['end_line']}: \"{r['quote']}\"")
        lines.append("")
    lines.append("Ownership is suggested for coordination only; it is not an accepted assignment.")
    return "\n".join(lines)
