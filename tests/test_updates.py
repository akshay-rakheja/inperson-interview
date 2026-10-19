from datetime import date

import pytest

from procurement.actions import build_owner_action_packs
from procurement.analysis import current_input_hash, run_case_analysis
from procurement.ingest import import_case
from procurement.models import CaseAnalysis
from procurement.store import Store
from procurement.updates import append_response, diff_analyses, mark_reviews_for_recheck

from conftest import ACME, NORTHSTAR
from fakes import FakeClient, LLMErrors, analysis, assessment, position, src_ref

AS_OF = date(2026, 10, 19)
MSGS = "03_internal/internal_messages.txt"
PEN = "05_evidence/pen_test_summary.txt"


def base_store(cases_copy, tmp_path):
    store = Store(tmp_path / "w.sqlite")
    for cid in (ACME, NORTHSTAR):
        store.save_sources(cid, import_case(cases_copy, cid))
    return store


def before(sources):
    rem = position("Engineering reports remediation; evidence needed.",
                   [src_ref(sources, MSGS, 5, 5, "remediated later, but need evidence")])
    dpa_dep = {"description": "DPA blocked until subprocessors and cross-border transfers are confirmed",
               "basis": "explicit", "refs": [src_ref(sources, MSGS, 6, 6, "DPA blocked until subprocessors")]}
    return analysis(ACME, AS_OF.isoformat(), [
        assessment(ACME, "Q-008", title="Pen test", positions=[rem], gaps=["Closure evidence missing"],
                   next_action="Provide M-01/M-02 closure evidence"),
        assessment(ACME, "Q-020", title="DPA blockers", deps=[dpa_dep], gaps=["Subprocessors unconfirmed"],
                   next_action="Confirm subprocessors", owner="Mina (Legal)"),
        assessment(ACME, "Q-004", title="Retention", gaps=["Trace retention unknown", "Prompt retention unknown"],
                   next_action="Confirm per-data-type retention"),
    ])


def after(sources, response):
    data = before(sources)
    pen = data["assessments"][0]
    pen["current_position"].append(position(
        "Simulated demo input: Engineering reports a retest targeted for 2026-11-06.",
        [{"source_id": response.source_id, "start_line": 2, "end_line": 2, "quote": "retest targeted for 2026-11-06"}]))
    pen["commitments"] = [{"text": "External retest letter", "owner": "Anthony", "date_iso": "2026-11-06",
                           "date_text": "2026-11-06", "date_kind": "reported_owner_commitment",
                           "refs": [{"source_id": response.source_id, "start_line": 2, "end_line": 2,
                                     "quote": "retest targeted for 2026-11-06"}]}]
    return data


def test_response_is_immutable_deduplicated_case_isolated_and_stales_drafts(cases_copy, tmp_path):
    store = base_store(cases_copy, tmp_path)
    store.save_draft(ACME, "run-x", {"generated": {"text": "t"}})
    store.save_draft(NORTHSTAR, "run-y", {"generated": {"text": "t"}})
    kwargs = dict(case_id=ACME, text="M-01 retest targeted for 2026-11-06.", label="demo retest",
                  reported_author="Anthony", simulated=True, related_keys=[f"{ACME}:Q-008"], store=store)
    first = append_response(**kwargs)
    again = append_response(**kwargs)
    assert first.source_id == again.source_id
    assert sum(s.kind == "response" for s in store.get_source_versions(ACME)) == 1
    assert not any(s.kind == "response" for s in store.get_sources(NORTHSTAR))
    assert first.text.startswith("[SIMULATED DEMO INPUT] demo retest")
    assert first.metadata["author_verified"] is False and first.metadata["simulated"] is True
    assert store.get_drafts(ACME)[0]["stale"] and not store.get_drafts(NORTHSTAR)[0]["stale"]
    with pytest.raises(ValueError):
        append_response(**{**kwargs, "related_keys": [f"{NORTHSTAR}:Q-008"]})
    # Originals are untouched on disk.
    assert not (cases_copy / "CASE_1_Acme_Bank" / "responses").exists()


def test_failed_refresh_keeps_old_run_visible_as_outdated_and_retry_does_not_duplicate(cases_copy, tmp_path):
    store = base_store(cases_copy, tmp_path)
    sources = store.get_sources(ACME)
    ok = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF, client=FakeClient([before(sources)]))["run"]
    resp = append_response(case_id=ACME, text="Retest booked\nretest targeted for 2026-11-06", label="demo",
                           reported_author="Anthony", simulated=True, related_keys=[], store=store)
    failing = FakeClient([LLMErrors.timeout])
    failed = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF, client=failing)["run"]
    assert failed["status"] == "failed"
    latest = store.latest_run(ACME)
    assert latest["run_id"] == ok["run_id"]
    assert latest["input_hash"] != current_input_hash(store, ACME, AS_OF, failing)  # shown as outdated
    # Owner packs still derive from the displayed (old) run without a model call.
    packs = build_owner_action_packs(analysis=CaseAnalysis.model_validate(latest["analysis"]))
    assert packs and len(failing.calls) == 1

    retry = FakeClient([after(store.get_sources(ACME), resp)])
    new = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF, client=retry)["run"]
    assert new["status"] == "success" and new["previous_run_id"] == ok["run_id"]
    assert sum(s.kind == "response" for s in store.get_source_versions(ACME)) == 1


def test_diff_reports_changed_commitment_and_unchanged_blockers(cases_copy, tmp_path):
    store = base_store(cases_copy, tmp_path)
    sources = store.get_sources(ACME)
    resp = append_response(case_id=ACME, text="Retest booked\nretest targeted for 2026-11-06", label="demo",
                           reported_author="Anthony", simulated=True, related_keys=[], store=store)
    b = CaseAnalysis.model_validate(before(sources))
    a = CaseAnalysis.model_validate(after(store.get_sources(ACME), resp))
    diff = diff_analyses(b, a)
    pen = diff["changed"][f"{ACME}:Q-008"]
    assert "commitments" in pen and "current_position" in pen and "evidence_state" not in pen
    assert pen["commitments"]["added"][0].endswith("[reported_owner_commitment; 2026-11-06]")
    assert f"{ACME}:Q-020" in diff["unchanged_open"] and f"{ACME}:Q-004" in diff["unchanged_open"]
    assert f"{ACME}:Q-008" in diff["still_open_after"]  # closure evidence still missing


def test_added_and_removed_keys_are_reported_not_merged():
    ref = {"source_id": f"{ACME}:x@1", "start_line": 1, "end_line": 1, "quote": "q"}
    b = CaseAnalysis.model_validate(analysis(ACME, "2026-10-19", [assessment(ACME, "Q-001", gaps=["g"])]))
    a = CaseAnalysis.model_validate(analysis(ACME, "2026-10-19", [assessment(ACME, "Q-002", gaps=["g"])]))
    diff = diff_analyses(b, a)
    assert diff["added"] == [f"{ACME}:Q-002"] and diff["removed"] == [f"{ACME}:Q-001"]


def test_partial_answer_keeps_other_owner_asks(cases_copy, tmp_path):
    """An app-log answer must not remove the separate trace/prompt retention asks."""
    store = base_store(cases_copy, tmp_path)
    data = before(store.get_sources(ACME))
    data["assessments"][2]["gaps"] = ["Trace retention unknown", "Prompt retention unknown"]
    packs = build_owner_action_packs(analysis=CaseAnalysis.model_validate(data))
    items = [i for p in packs for i in p["items"] if i["requirement_key"] == f"{ACME}:Q-004"]
    assert items and items[0]["gaps"] == ["Trace retention unknown", "Prompt retention unknown"]


def test_reviews_marked_for_recheck_and_owner_overrides_persist(cases_copy, tmp_path):
    store = base_store(cases_copy, tmp_path)
    sources = store.get_sources(ACME)
    store.save_review(ACME, f"{ACME}:Q-008", {"review_note": "verified wording", "reviewer": "Jai",
                                              "owner_override": "Anthony"})
    store.save_review(ACME, f"{ACME}:Q-020", {"owner_override": "Mina"})
    store.save_review(ACME, f"{ACME}:Q-099", {"review_note": "old item"})
    new = CaseAnalysis.model_validate(before(sources))
    assert mark_reviews_for_recheck(store, ACME, "run-new", new) == 2
    assert mark_reviews_for_recheck(store, ACME, "run-new", new) == 0  # idempotent per run
    reviews = {r["requirement_key"]: r for r in store.get_reviews(ACME)}
    assert reviews[f"{ACME}:Q-008"]["needs_recheck"] and reviews[f"{ACME}:Q-008"]["owner_override"] == "Anthony"
    assert "no longer present" in reviews[f"{ACME}:Q-099"]["recheck_reason"]
    assert "needs_recheck" not in reviews[f"{ACME}:Q-020"]
