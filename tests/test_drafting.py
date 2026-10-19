from datetime import date

import pytest

from procurement.analysis import run_case_analysis
from procurement.drafting import build_draft_payload, check_update, create_draft, save_edit, save_local_review
from procurement.ingest import import_case
from procurement.llm import LLMError
from procurement.models import CaseAnalysis, CustomerUpdate
from procurement.store import Store
from procurement.updates import append_response

from conftest import ACME
from fakes import FakeClient, LLMErrors, analysis, assessment, position, src_ref

AS_OF = date(2026, 10, 19)
PEN = "05_evidence/pen_test_summary.txt"
MSGS = "03_internal/internal_messages.txt"


def setup(cases_copy, tmp_path):
    store = Store(tmp_path / "w.sqlite")
    sources = import_case(cases_copy, ACME)
    store.save_sources(ACME, sources)
    a = analysis(ACME, AS_OF.isoformat(), [assessment(
        ACME, "Q-008", title="Pen test",
        positions=[position("Engineering reports remediation; evidence needed.",
                            [src_ref(sources, MSGS, 5, 5, "remediated later, but need evidence")])],
        gaps=["Closure evidence missing"], next_action="Provide closure evidence")])
    run = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF, client=FakeClient([a]))["run"]
    return store, sources, run


def update_dict(sources, quote="remediated later, but need evidence", marker=True):
    return {
        "text": "Engineering reports remediation is complete; closure evidence is pending." + (" [C1]" if marker else ""),
        "claims": [{"statement": "Engineering reports remediation is complete; closure evidence is pending.",
                    "refs": [src_ref(sources, MSGS, 5, 5, quote)]}],
        "unresolved_items": ["M-01/M-02 closure evidence", "Retest date unconfirmed"],
    }


def test_draft_persists_basis_run_metadata_and_checks(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    draft_id = create_draft(store=store, case_id=ACME, run=run, selected_keys=[f"{ACME}:Q-008"],
                            client=FakeClient([update_dict(sources)]))
    d = store.get_drafts(ACME)[0]
    assert d["draft_id"] == draft_id and d["basis_run_id"] == run["run_id"] and not d["stale"]
    assert d["generation"]["fake"] is True and d["generation"]["prompt_version"] == "draft-v1"
    assert d["claim_checks"][0]["status"] == "ok" and d["validation_warnings"] == []
    assert "Retest date unconfirmed" in d["generated"]["unresolved_items"]


def test_invented_citation_marked_unsupported(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    create_draft(store=store, case_id=ACME, run=run, selected_keys=[f"{ACME}:Q-008"],
                 client=FakeClient([update_dict(sources, quote="retest confirmed closure")]))
    d = store.get_drafts(ACME)[0]
    assert d["claim_checks"][0]["status"] == "unsupported"
    assert d["generated"]["claims"][0]["refs"] == []


def test_missing_claim_marker_warns(cases_copy):
    sources = import_case(cases_copy, ACME)
    _, _, warnings = check_update(CustomerUpdate.model_validate(update_dict(sources, marker=False)), sources, ACME)
    assert any("not referenced" in w for w in warnings)


def test_unknown_selected_key_rejected(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    with pytest.raises(ValueError):
        build_draft_payload(analysis=CaseAnalysis.model_validate(run["analysis"]), sources=sources,
                            selected_keys=[f"{ACME}:Q-999"], reviews=[])


def test_payload_includes_original_excerpts_and_marks_review_as_context(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    reviews = [{"requirement_key": f"{ACME}:Q-008", "review_note": "looks right", "reviewer": "Jai"}]
    payload = build_draft_payload(analysis=CaseAnalysis.model_validate(run["analysis"]), sources=sources,
                                  selected_keys=[f"{ACME}:Q-008"], reviews=reviews)
    assert "5| Anthony: Feb pen test had 2 mediums" in payload["user"]
    assert "self-reported by Jai" in payload["user"] and "not approval" in payload["user"]


def test_provider_failure_keeps_prior_drafts(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    create_draft(store=store, case_id=ACME, run=run, selected_keys=[f"{ACME}:Q-008"],
                 client=FakeClient([update_dict(sources)]))
    with pytest.raises(LLMError):
        create_draft(store=store, case_id=ACME, run=run, selected_keys=[f"{ACME}:Q-008"],
                     client=FakeClient([LLMErrors.credits]))
    assert len(store.get_drafts(ACME)) == 1


def test_edits_and_local_review_survive_reload_and_are_labelled(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    draft_id = create_draft(store=store, case_id=ACME, run=run, selected_keys=[f"{ACME}:Q-008"],
                            client=FakeClient([update_dict(sources)]))
    d = store.get_drafts(ACME)[0]
    save_edit(store, draft_id, "Edited v1", d)
    save_edit(store, draft_id, "Edited v2", Store(tmp_path / "w.sqlite").get_drafts(ACME)[0])
    save_local_review(store, draft_id, "Jai", "OK to share after Legal check")
    d = Store(tmp_path / "w.sqlite").get_drafts(ACME)[0]
    assert d["edited_text"] == "Edited v2" and d["edit_history"][0]["text"] == "Edited v1"
    assert d["grounding"] == "edited text: grounding not rechecked"
    assert d["generated"]["text"].startswith("Engineering reports")  # generated text untouched
    assert "self-reported" in d["review"]["label"] and "not customer acceptance" in d["review"]["label"]
    with pytest.raises(ValueError):
        store.update_draft(draft_id, {"generated": {"text": "overwrite"}})


def test_new_source_makes_draft_stale(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    create_draft(store=store, case_id=ACME, run=run, selected_keys=[f"{ACME}:Q-008"],
                 client=FakeClient([update_dict(sources)]))
    append_response(case_id=ACME, text="Retest booked.", label="demo", reported_author="Anthony", simulated=True,
                    related_keys=[], store=store)
    d = store.get_drafts(ACME)[0]
    assert d["stale"] and "New information added" in d["stale_reason"]


def test_simulated_citation_must_be_labelled(cases_copy, tmp_path):
    store, sources, run = setup(cases_copy, tmp_path)
    rec = append_response(case_id=ACME, text="Retest letter targeted for 2026-11-06.", label="demo",
                          reported_author="Anthony", simulated=True, related_keys=[], store=store)
    upd = CustomerUpdate.model_validate({
        "text": "A retest letter is targeted for 2026-11-06. [C1]",
        "claims": [{"statement": "A retest letter is targeted for 2026-11-06.",
                    "refs": [{"source_id": rec.source_id, "start_line": 2, "end_line": 2,
                              "quote": "Retest letter targeted for 2026-11-06."}]}],
        "unresolved_items": []})
    _, checks, warnings = check_update(upd, store.get_sources(ACME), ACME)
    assert checks[0]["status"] == "check" and checks[0]["cites_simulated"]
    assert any("simulated" in w for w in warnings)
