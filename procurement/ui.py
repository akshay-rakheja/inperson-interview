"""Streamlit workspace. Model requests happen only inside explicit button handlers.

All source text is rendered with st.code/st.text (never as HTML or executed).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

from . import config
from .actions import build_owner_action_packs, pack_as_text
from .analysis import current_input_hash, input_hash, input_identity, run_case_analysis
from .drafting import create_draft, save_edit, save_local_review
from .ingest import discover_cases, questionnaire_rows, scan_case
from .llm import LLMError, ModelClient
from .models import CaseAnalysis, EvidenceRef, SourceRecord
from .store import Store
from .updates import append_response, diff_analyses, mark_reviews_for_recheck

FIXTURE_DIR = config.REPO_ROOT / "tests" / "fixtures"
STATE_BADGE = {"found": "🟢 found", "partial": "🟡 partial", "missing": "🔴 missing", "conflicting": "🟣 conflicting"}


def toronto_today():
    return datetime.now(ZoneInfo("America/Toronto")).date()


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------


def render_sources(refs: list[EvidenceRef], sources: dict[str, SourceRecord], *, context: int = 1) -> None:
    for ref in refs:
        if isinstance(ref, dict):
            ref = EvidenceRef.model_validate(ref)
        src = sources.get(ref.source_id)
        if src is None:
            st.error(f"Missing source {ref.source_id} (not in this case's current sources)")
            continue
        lines = src.lines()
        lo, hi = max(1, ref.start_line - context), min(len(lines), ref.end_line + context)
        tag = ""
        if src.kind == "response":
            tag = " · SIMULATED DEMO INPUT" if src.metadata.get("simulated") else " · pasted response (author unverified)"
        st.caption(f"`{src.relative_path}` L{ref.start_line}-{ref.end_line}{tag} · quote: “{ref.quote}”")
        numbered = "\n".join(
            f"{'▶' if ref.start_line <= i <= ref.end_line else ' '} {i:>3}| {lines[i - 1]}" for i in range(lo, hi + 1)
        )
        st.code(numbered, language=None)


def _position_line(p) -> str:
    date = p.effective_date_iso or p.date_text
    date_part = f" · date: {date}" if date else ""
    return f"**{p.kind}** · {p.applicability.replace('_', ' ')}{date_part} — {p.text}"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _sidebar(client: ModelClient | None) -> tuple[dict, object, bool]:
    settings = config.get_settings()
    cases = discover_cases(config.REPO_ROOT)
    st.sidebar.header("Case")
    case = st.sidebar.selectbox("Case folder", cases, format_func=lambda c: f"{c['folder']} — {c['title'][:40]}")
    as_of = st.sidebar.date_input("Assessment date (America/Toronto)", value=toronto_today(), key="as_of")
    st.sidebar.caption(f"Assessing as of **{as_of.isoformat()}**. Expiry interpretation depends on this date.")
    force = st.sidebar.checkbox("Force a fresh model run (ignore cached result)", value=False)
    st.sidebar.divider()
    st.sidebar.subheader("Model configuration")
    if settings.has_key:
        st.sidebar.success(f"Provider: OpenRouter\n\nModel: `{settings.model}`")
    else:
        st.sidebar.warning(settings.status_message)
    st.sidebar.caption(f"Endpoint `{settings.base_url}` · key file `{settings.env_path}` · no automatic model "
                       "or provider fallback.")
    count = getattr(client, "request_count", 0) if client else 0
    st.sidebar.metric("Model requests this session", count)
    return case, as_of, force


def _ensure_sources(store: Store, case_id: str) -> list[str]:
    records, warnings = scan_case(config.REPO_ROOT, case_id)
    store.save_sources(case_id, records)  # idempotent; changed originals become new retained versions
    return warnings


def _run_label(run: dict) -> str:
    kind = "FAKE/test client" if run.get("fake") else "live model"
    meta = run.get("request_metadata") or {}
    returned = meta.get("returned_model")
    return (f"`{run['run_id']}` · {run['created_at']} · as of {run['as_of_date']} · {kind} · "
            f"{run['provider']} `{run['model']}`" + (f" (returned `{returned}`)" if returned else "")
            + f" · prompt {run['prompt_version']}")


def _do_analysis(store: Store, case_id: str, as_of, client: ModelClient, force: bool) -> None:
    with st.spinner("Running one full-case analysis request… (this can take a few minutes)"):
        result = run_case_analysis(store=store, case_id=case_id, as_of_date=as_of, client=client, force=force)
    run = result["run"]
    if result["reused"]:
        st.session_state["flash"] = ("info", f"Reused saved run {run['run_id']} for identical inputs; no request made.")
    elif run["status"] == "success":
        rechecked = 0
        if run.get("previous_run_id"):
            rechecked = mark_reviews_for_recheck(store, case_id, run["run_id"],
                                                 CaseAnalysis.model_validate(run["analysis"]))
        st.session_state["flash"] = ("success", f"Analysis run {run['run_id']} succeeded."
                                     + (f" {rechecked} review note(s) marked for recheck." if rechecked else ""))
    else:
        st.session_state["flash"] = ("error", f"Analysis failed; the previous successful run is still shown. "
                                     f"Error: {run['error']}")


def render_workspace(store: Store, client: ModelClient | None) -> None:
    case, as_of, force = _sidebar(client)
    case_id = case["case_id"]
    import_warnings = _ensure_sources(store, case_id)
    sources = store.get_sources(case_id)
    sources_by_id = {s.source_id: s for s in sources}

    st.title("Procurement evidence workspace")
    st.markdown(f"**{case['title']}** · case `{case_id}` · {len(sources)} current sources "
                f"({sum(s.kind == 'response' for s in sources)} pasted responses)")
    for w in import_warnings:
        st.warning(w)

    run = store.latest_run(case_id)
    runs = store.list_runs(case_id)
    latest_failed = runs[0] if runs and runs[0]["status"] == "failed" else None

    if client is not None:
        current_hash = current_input_hash(store, case_id, as_of, client)
    else:
        current_hash = input_hash(input_identity(sources=sources, as_of_date=as_of, client=type(
            "C", (), {"provider": config.PROVIDER, "base_url": config.OPENROUTER_BASE_URL,
                      "model": config.get_settings().model})()))
    outdated = bool(run and run["input_hash"] != current_hash)

    cols = st.columns([3, 1])
    with cols[0]:
        if run:
            st.markdown("Displayed analysis: " + _run_label(run))
            if outdated:
                reasons = []
                if run["as_of_date"] != as_of.isoformat():
                    reasons.append(f"assessment date differs ({run['as_of_date']} vs {as_of.isoformat()})")
                if set(run["source_ids"]) != {s.source_id for s in sources}:
                    reasons.append("sources changed since this run")
                if run["model"] != config.get_settings().model:
                    reasons.append("configured model changed")
                st.warning("**OUTDATED** — this run predates the current inputs: " + "; ".join(reasons or ["inputs differ"])
                           + ". Owner requests and drafts below are based on it until a refresh succeeds.")
            else:
                st.success("Analysis covers the current sources, assessment date and model.")
        else:
            st.info("No successful analysis yet for this case. Run one explicitly.")
        if latest_failed:
            st.error(f"Latest attempt `{latest_failed['run_id']}` failed at {latest_failed['created_at']}: "
                     f"{latest_failed['error']}")
    with cols[1]:
        disabled = client is None
        if st.button("Run analysis", type="primary", disabled=disabled, key=f"run-{case_id}",
                     help="One explicit model request (or reuse of an identical saved run)."):
            _do_analysis(store, case_id, as_of, client, force)
            st.rerun()
    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])

    analysis = CaseAnalysis.model_validate(run["analysis"]) if run else None
    tab1, tab2, tab3 = st.tabs(["Requirements & gaps", "Customer update", "New information"])
    with tab1:
        _requirements_tab(store, case_id, run, analysis, sources, sources_by_id, outdated)
    with tab2:
        _customer_update_tab(store, case_id, run, analysis, sources_by_id, outdated, client)
    with tab3:
        _new_information_tab(store, case_id, run, analysis, sources, sources_by_id, as_of, client, force, outdated)


# ---------------------------------------------------------------------------
# Tab 1: requirements, owner packs, questionnaire, sources
# ---------------------------------------------------------------------------


def _requirements_tab(store, case_id, run, analysis, sources, sources_by_id, outdated):
    rows = questionnaire_rows(sources)
    reviews = {r["requirement_key"]: r for r in store.get_reviews(case_id)}
    if analysis is None:
        st.info("Run analysis to see grouped requirements. Original questionnaire and sources remain available below.")
    else:
        st.subheader("Summary")
        st.write(analysis.summary)
        st.caption("Evidence states describe what the sources support. They are not approvals, verification or "
                   "compliance verdicts. Citation checks prove a quote exists, not that it supports the conclusion.")
        if analysis.warnings:
            with st.expander(f"Warnings and provenance checks ({len(analysis.warnings)})"):
                for w in analysis.warnings:
                    st.text(f"• {w}")

        overrides = {k: r.get("owner_override", "") for k, r in reviews.items() if r.get("owner_override")}
        packs = build_owner_action_packs(analysis=analysis, owner_overrides=overrides)
        with st.expander(f"Actions by suggested owner ({len(packs)} owners)", expanded=False):
            st.caption("Deterministic grouping of the displayed run — no model call, nothing is sent. "
                       "Ownership is a suggestion or local edit, not an accepted assignment."
                       + (" **Based on an OUTDATED run.**" if outdated else ""))
            for pack in packs:
                st.markdown(f"#### {pack['owner_label']} ({len(pack['items'])} requests)")
                for item in pack["items"]:
                    basis = {"local_override": "locally edited owner", "suggested": "suggested owner",
                             "unassigned": "no owner suggested"}[item["owner_basis"]]
                    st.markdown(f"- **{item['title']}** `{item['requirement_key']}` · "
                                f"{STATE_BADGE[item['evidence_state']]} · {basis}\n\n  Ask: {item['ask']}")
                st.text_area("Copyable request text", pack_as_text(pack, case_id=case_id, run_id=run["run_id"],
                                                                 outdated=outdated),
                             height=180, key=f"pack-{case_id}-{pack['owner_label']}")

        st.subheader(f"Requirements ({len(analysis.assessments)})")
        f1, f2 = st.columns(2)
        states = f1.multiselect("Evidence state", list(STATE_BADGE), default=[], key=f"fs-{case_id}")
        workstreams = sorted({a.workstream for a in analysis.assessments})
        ws = f2.multiselect("Workstream", workstreams, default=[], key=f"fw-{case_id}")
        rows_by_id = {r["id"]: r for r in rows}
        for a in analysis.assessments:
            if states and a.evidence_state not in states or ws and a.workstream not in ws:
                continue
            review = reviews.get(a.requirement_key, {})
            owner = review.get("owner_override") or a.suggested_owner or "Unassigned"
            flag = " · ⚠ review needs recheck" if review.get("needs_recheck") else ""
            with st.expander(f"{STATE_BADGE[a.evidence_state]} · {a.title} · {owner}{flag}"):
                _render_assessment(store, case_id, a, review, rows_by_id, sources_by_id)

    with st.expander("Original questionnaire (all rows, original statuses)"):
        covered = {}
        if analysis:
            for a in analysis.assessments:
                for q in a.question_ids:
                    covered.setdefault(q, []).append(a.requirement_key)
            if analysis.unassessed_question_ids:
                st.warning(f"Unassessed rows ({len(analysis.unassessed_question_ids)}): "
                           + ", ".join(analysis.unassessed_question_ids))
        st.dataframe([
            {"id": r["id"], "section": r["section"], "question": r["question"],
             "original draft_answer": r["draft_answer"], "original status": r["status"], "note": r["note"],
             "assessed under": ", ".join(covered.get(r["id"], [])) or "—"}
            for r in rows
        ], use_container_width=True, hide_index=True)

    with st.expander("Browse sources"):
        choice = st.selectbox("Source", sources, format_func=lambda s: f"{s.relative_path} ({s.kind})",
                              key=f"src-{case_id}")
        if choice:
            st.caption(f"`{choice.source_id}` · sha256 {choice.content_sha256[:16]}… · metadata "
                       f"{json.dumps({k: v for k, v in choice.metadata.items() if k != 'related_keys'})}")
            st.code("\n".join(f"{i:>3}| {line}" for i, line in enumerate(choice.lines(), start=1)), language=None)


def _render_assessment(store, case_id, a, review, rows_by_id, sources_by_id):
    st.markdown(f"`{a.requirement_key}` · origin **{a.origin.replace('_', ' ')}** · workstream {a.workstream}")
    if a.question_ids:
        st.markdown("**Original questions**")
        for q in a.question_ids:
            r = rows_by_id.get(q)
            if r:
                extra = f" · original draft: “{r['draft_answer']}”" if r["draft_answer"] else ""
                st.text(f"{q} [{r['section']}] original status {r['status']}{extra}: {r['question'][:160]}")
    st.markdown("**Current position**")
    if not a.current_position:
        st.text("No source-backed position.")
    for p in a.current_position:
        st.markdown("- " + _position_line(p))
        render_sources(p.refs, sources_by_id)
    if a.gaps:
        st.markdown("**Gaps**")
        for g in a.gaps:
            st.markdown(f"- {g}")
    if a.dependencies:
        st.markdown("**Dependencies**")
        for d in a.dependencies:
            st.markdown(f"- *{d.basis}*: {d.description}")
            render_sources(d.refs, sources_by_id)
    st.markdown("**Commitments (source-reported only)**")
    if not a.commitments:
        st.text("None recorded in sources.")
    for c in a.commitments:
        when = c.date_iso or (f"“{c.date_text}” (exact date unconfirmed)" if c.date_text else "date unconfirmed")
        st.markdown(f"- {c.text} · {c.date_kind.replace('_', ' ')} · {when} · owner: {c.owner or 'not stated'}")
        render_sources(c.refs, sources_by_id)
    st.markdown(f"**Next action:** {a.next_action or '—'}")
    st.markdown(f"**Suggested owner:** {a.suggested_owner or 'Unassigned'} *(suggestion only)*")

    with st.form(key=f"review-{a.requirement_key}"):
        st.caption("Local overlay. Self-reported; not authenticated approval or customer acceptance.")
        owner = st.text_input("Owner override (coordination choice)", value=review.get("owner_override", ""))
        reviewer = st.text_input("Your name", value=review.get("reviewer", ""))
        note = st.text_area("Review note", value=review.get("review_note", ""))
        if review.get("needs_recheck"):
            st.warning(f"Prior review needs recheck: {review.get('recheck_reason')} "
                       f"(run {review.get('recheck_against_run')})")
        if st.form_submit_button("Save overlay"):
            fields = {"owner_override": owner.strip()}
            if note.strip() != review.get("review_note", "") or reviewer.strip() != review.get("reviewer", ""):
                fields.update({"review_note": note.strip(), "reviewer": reviewer.strip(),
                               "reviewed_at": datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds"),
                               "review_label": "self-reported", "needs_recheck": False})
            store.save_review(case_id, a.requirement_key, fields)
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: customer update drafts
# ---------------------------------------------------------------------------


def _customer_update_tab(store, case_id, run, analysis, sources_by_id, outdated, client):
    st.caption("Drafts are generated only when you click Generate, are never sent anywhere, and must be reviewed. "
               "Citation checks verify quotes exist; inspect each claim against its source.")
    if analysis is None:
        st.info("Run analysis first.")
    else:
        options = [a.requirement_key for a in analysis.assessments]
        titles = {a.requirement_key: f"{a.title} ({a.evidence_state})" for a in analysis.assessments}
        selected = st.multiselect("Findings to include", options, format_func=lambda k: titles[k],
                                  key=f"sel-{case_id}")
        if outdated:
            st.warning("The displayed analysis is outdated. Reanalyse before generating a new draft.")
        if st.button("Generate customer update", disabled=client is None or not selected or outdated,
                     key=f"gen-{case_id}"):
            try:
                with st.spinner("Generating draft (one model request)…"):
                    draft_id = create_draft(store=store, case_id=case_id, run=run, selected_keys=selected,
                                            client=client)
                st.session_state["flash"] = ("success", f"Draft {draft_id} generated.")
            except (LLMError, ValueError) as exc:
                st.session_state["flash"] = ("error", f"Draft generation failed; earlier drafts kept. {exc}")
            st.rerun()

    drafts = store.get_drafts(case_id)
    st.subheader(f"Drafts ({len(drafts)})")
    for d in drafts:
        not_current = d["stale"] or not run or d["basis_run_id"] != run["run_id"]
        status = "STALE" if d["stale"] else ("NOT CURRENT (different basis run)" if not_current else "current draft")
        gen = d["generation"]
        with st.expander(f"{status} · {d['draft_id']} · {gen['generated_at']} · basis {d['basis_run_id']}",
                         expanded=not not_current and d is drafts[0]):
            if d["stale"]:
                st.error(f"Stale: {d['stale_reason']}")
            st.caption(f"{'FAKE/test client' if gen.get('fake') else 'live model'} · {gen['provider']} "
                       f"`{gen['model']}` · prompt {gen['prompt_version']} · findings: {', '.join(d['selected_keys'])}")
            generated = d["generated"]
            if d.get("edited_text") is not None:
                st.warning("Edited text — grounding not rechecked after human edits.")
            text = st.text_area("Draft text (editable)", value=d.get("edited_text") or generated["text"], height=320,
                                key=f"edit-{d['draft_id']}")
            c1, c2 = st.columns(2)
            if c1.button("Save edit", key=f"save-{d['draft_id']}"):
                save_edit(store, d["draft_id"], text, d)
                st.rerun()
            header = (f"[{status.upper()} · DRAFT NOT APPROVED · case {case_id} · basis run {d['basis_run_id']}"
                      f"{' · edited, grounding not rechecked' if d.get('edited_text') is not None else ''}]\n\n")
            c2.download_button("Download draft (.txt)", header + text, file_name=f"{d['draft_id']}.txt",
                               key=f"dl-{d['draft_id']}")
            if d.get("validation_warnings"):
                st.warning("Deterministic checks: " + " | ".join(d["validation_warnings"]))
            st.markdown("**Claims and citations**")
            for claim, check in zip(generated["claims"], d.get("claim_checks", [])):
                icon = {"ok": "✅ quote verified", "check": "⚠ check", "unsupported": "❌ unsupported"}[check["status"]]
                st.markdown(f"- **{check['claim']}** {icon}: {claim['statement']}")
                render_sources(claim["refs"], sources_by_id, context=0)
            st.markdown("**Unresolved items**")
            for u in generated["unresolved_items"]:
                st.markdown(f"- {u}")
            review = d.get("review")
            if review:
                st.info(f"Local review by {review['reviewer']} at {review['reviewed_at']}: {review['note']} "
                        f"— {review['label']}")
            with st.form(key=f"rev-{d['draft_id']}"):
                reviewer = st.text_input("Reviewer name (self-reported)")
                note = st.text_area("Review note")
                if st.form_submit_button("Record local review"):
                    if reviewer.strip():
                        save_local_review(store, d["draft_id"], reviewer, note)
                        st.rerun()
                    else:
                        st.warning("Enter a reviewer name.")


# ---------------------------------------------------------------------------
# Tab 3: new information, reanalysis comparison, run history
# ---------------------------------------------------------------------------


def _load_fixtures(case_id: str) -> list[dict]:
    out = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("case_id") == case_id:
            out.append({**data, "_file": path.name})
    return out


def _new_information_tab(store, case_id, run, analysis, sources, sources_by_id, as_of, client, force, outdated):
    st.caption("Pasted information becomes a new immutable source. Existing drafts become stale immediately. "
               "Reanalysis is a separate explicit action. Authors are entered, not verified.")
    fixtures = _load_fixtures(case_id)
    k = lambda name: f"ni-{case_id}-{name}"  # noqa: E731
    if fixtures:
        fx = st.selectbox("Prefill from a clearly-labelled demo fixture", [None, *fixtures],
                          format_func=lambda f: "—" if f is None else f["label"], key=k("fixture"))
        if st.button("Load fixture into form", key=k("loadfx")) and fx:
            st.session_state[k("label")] = fx["label"]
            st.session_state[k("author")] = fx.get("reported_author", "")
            st.session_state[k("sim")] = bool(fx.get("simulated", True))
            st.session_state[k("eff")] = fx.get("effective_date_text", "")
            st.session_state[k("text")] = fx["text"]
            if analysis:
                keys = {a.requirement_key for a in analysis.assessments}
                st.session_state[k("rel")] = [
                    a.requirement_key for a in analysis.assessments
                    if set(a.question_ids) & set(fx.get("related_question_ids", []))] if keys else []
            st.rerun()

    with st.form(key=k("form")):
        label = st.text_input("Label", key=k("label"))
        author = st.text_input("Reported author (unverified)", key=k("author"))
        simulated = st.checkbox("Simulated demo input", value=True, key=k("sim"))
        eff = st.text_input("Effective date as stated (optional, free text)", key=k("eff"))
        options = [a.requirement_key for a in analysis.assessments] if analysis else []
        rel = st.multiselect("Related requirements (optional)", options, key=k("rel"))
        text = st.text_area("Response text", height=180, key=k("text"))
        if st.form_submit_button("Save as new source"):
            try:
                rec = append_response(case_id=case_id, text=text, label=label, reported_author=author or None,
                                      simulated=simulated, related_keys=rel, store=store,
                                      effective_date_text=eff or None)
                st.session_state["flash"] = ("success", f"Saved {rec.source_id}. Existing drafts are now stale. "
                                             "Click Reanalyse to refresh the assessment.")
            except ValueError as exc:
                st.session_state["flash"] = ("error", str(exc))
            st.rerun()

    if st.button("Reanalyse with current sources", type="primary", disabled=client is None, key=k("rerun")):
        _do_analysis(store, case_id, as_of, client, force)
        st.rerun()

    responses = [s for s in sources if s.kind == "response"]
    st.subheader(f"Pasted responses ({len(responses)})")
    for s in responses:
        badge = "SIMULATED DEMO INPUT" if s.metadata.get("simulated") else "pasted response"
        st.markdown(f"- **{s.metadata.get('label')}** · {badge} · author (unverified): "
                    f"{s.metadata.get('reported_author') or 'unknown'} · received {s.metadata.get('received_at')} · "
                    f"`{s.source_id}`")

    st.subheader("What changed in the latest refresh")
    if not run or not run.get("previous_run_id"):
        st.info("No earlier successful run to compare with yet.")
    else:
        prev = store.get_run(run["previous_run_id"])
        if prev and prev["status"] == "success":
            if outdated:
                st.warning("A newer source or setting is not yet reflected; this comparison is between older runs.")
            _render_diff(diff_analyses(CaseAnalysis.model_validate(prev["analysis"]), analysis), prev, run)

    with st.expander("Run history"):
        st.dataframe([
            {"run": r["run_id"], "status": r["status"], "created": r["created_at"], "as_of": r.get("as_of_date"),
             "model": r.get("model"), "fake": r.get("fake"), "sources": len(r.get("source_ids") or []),
             "previous": r.get("previous_run_id"), "error": (r.get("error") or "")[:200]}
            for r in store.list_runs(case_id)
        ], use_container_width=True, hide_index=True)


def _render_diff(diff: dict, prev: dict, run: dict) -> None:
    st.caption(f"Comparing `{prev['run_id']}` → `{run['run_id']}` by stable requirement key. Rewording appears "
               "as change; this is not a semantic diff.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Changed requirements", len(diff["changed"]))
    c2.metric("Added / removed", f"{len(diff['added'])} / {len(diff['removed'])}")
    c3.metric("Still open after refresh", len(diff["still_open_after"]))
    for key in diff["added"]:
        st.markdown(f"- ➕ **added** `{key}`")
    for key in diff["removed"]:
        st.markdown(f"- ➖ **removed** `{key}` (review whether it was merged or dropped)")
    for key, fields in diff["changed"].items():
        with st.expander(f"Changed: {key} ({', '.join(fields)})"):
            for name, delta in fields.items():
                st.markdown(f"**{name}**")
                if "added" in delta:
                    for x in delta["removed"]:
                        st.text(f"  − {x}")
                    for x in delta["added"]:
                        st.text(f"  + {x}")
                else:
                    st.text(f"  before: {delta['before']}\n  after:  {delta['after']}")
    if diff["unchanged_open"]:
        st.markdown("**Unchanged and still open:** " + ", ".join(f"`{k}`" for k in diff["unchanged_open"]))
