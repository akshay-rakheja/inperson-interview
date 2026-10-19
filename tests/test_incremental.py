"""Incremental reanalysis (#14): requirements untouched by new sources are carried forward verbatim."""

from datetime import date

from procurement.analysis import plan_reanalysis, run_case_analysis
from procurement.ingest import import_case
from procurement.models import CaseAnalysis, SourceRecord
from procurement.store import Store
from procurement.updates import append_response, diff_analyses, mark_reviews_for_recheck

from conftest import ACME
from fakes import FakeClient, analysis, assessment, position, src_ref

AS_OF = date(2026, 10, 19)
MSGS = "03_internal/internal_messages.txt"
PEN_KEY, DPA_KEY, RET_KEY = f"{ACME}:Q-008", f"{ACME}:Q-020", f"{ACME}:Q-004"


def initial(sources):
    rem = position("Engineering reports remediation; evidence needed.",
                   [src_ref(sources, MSGS, 5, 5, "remediated later, but need evidence")])
    dpa = {"description": "DPA blocked until subprocessors and cross-border transfers are confirmed",
           "basis": "explicit", "refs": [src_ref(sources, MSGS, 6, 6, "DPA blocked until subprocessors")]}
    return analysis(ACME, AS_OF.isoformat(), [
        assessment(ACME, "Q-008", title="Pen test", positions=[rem], gaps=["Closure evidence missing"]),
        assessment(ACME, "Q-020", title="DPA blockers", deps=[dpa], gaps=["Subprocessors unconfirmed"]),
        assessment(ACME, "Q-004", title="Retention", gaps=["Trace retention unknown"]),
    ])


def seeded(cases_copy, tmp_path):
    store = Store(tmp_path / "w.sqlite")
    store.save_sources(ACME, import_case(cases_copy, ACME))
    run = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF,
                            client=FakeClient([initial(store.get_sources(ACME))]))["run"]
    return store, run


def respond(store, related, text="Retest booked\nretest targeted for 2026-11-06"):
    return append_response(case_id=ACME, text=text, label="demo retest", reported_author="Anthony",
                           simulated=True, related_keys=related, store=store)


def rerun_output(store, resp, *, include_pen=True, reword_carried=True):
    """What a model might return: the affected item updated, plus unrequested rewording of a carried item."""
    data = initial(store.get_sources(ACME))
    pen, dpa, ret = data["assessments"]
    pen["current_position"].append(position(
        "Simulated demo input: Engineering reports a retest targeted for 2026-11-06.",
        [{"source_id": resp.source_id, "start_line": 3, "end_line": 3, "quote": "retest targeted for 2026-11-06"}]))
    dpa["evidence_state"] = "missing"  # the kind of unprompted flip seen live on Northstar
    dpa["title"] = "DPA blockers (reworded)"
    out = ([pen] if include_pen else []) + ([dpa] if reword_carried else [])
    return analysis(ACME, AS_OF.isoformat(), out)


def plan(store, prev, **kw):
    client = kw.pop("client", FakeClient([]))
    return plan_reanalysis(previous_run=prev, sources=kw.pop("sources", store.get_sources(ACME)),
                           as_of_date=kw.pop("as_of", AS_OF), client=client, **kw)


# -- planning rules ---------------------------------------------------------


def test_no_previous_run_is_full(cases_copy, tmp_path):
    store = Store(tmp_path / "w.sqlite")
    store.save_sources(ACME, import_case(cases_copy, ACME))
    p = plan(store, None)
    assert p["mode"] == "full" and "no previous" in p["reason"]


def test_scoped_response_reanalyses_only_related_keys(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    resp = respond(store, [PEN_KEY])
    p = plan(store, prev)
    assert p["mode"] == "incremental"
    assert p["reanalyse"] == [PEN_KEY] and p["carry"] == [DPA_KEY, RET_KEY]
    assert p["new_source_ids"] == [resp.source_id]
    assert p["reasons"][PEN_KEY] == [f"related to new source {resp.source_id}"]


def test_unscoped_response_forces_full(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    respond(store, [])
    p = plan(store, prev)
    assert p["mode"] == "full" and "no related requirements" in p["reason"]


def test_related_key_unknown_to_previous_run_forces_full(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    respond(store, [f"{ACME}:Q-099"])
    p = plan(store, prev)
    assert p["mode"] == "full" and "Q-099" in p["reason"]


def test_changed_original_source_forces_full(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    respond(store, [PEN_KEY])
    msgs = next(s for s in store.get_sources(ACME) if s.relative_path == MSGS)
    edited = msgs.text + "New internal line.\n"
    store.save_sources(ACME, [SourceRecord(source_id=f"{ACME}:{MSGS}@edited00000", case_id=ACME,
                                           relative_path=MSGS, content_sha256="edited", text=edited,
                                           kind=msgs.kind, metadata=msgs.metadata)])
    p = plan(store, prev)
    assert p["mode"] == "full" and "changed or removed" in p["reason"]


def test_new_non_response_source_forces_full(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    store.save_sources(ACME, [SourceRecord(source_id=f"{ACME}:06_new/note.txt@abc", case_id=ACME,
                                           relative_path="06_new/note.txt", content_sha256="abc",
                                           text="A new case document.\n", kind="text", metadata={})])
    p = plan(store, prev)
    assert p["mode"] == "full" and "not a scoped response" in p["reason"]


def test_date_model_or_prompt_change_and_force_are_full(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    respond(store, [PEN_KEY])
    assert plan(store, prev, as_of=date(2026, 11, 1))["mode"] == "full"
    assert plan(store, prev, client=FakeClient([], model="other/model"))["mode"] == "full"
    assert plan(store, {**prev, "prompt_version": "analysis-v0"})["mode"] == "full"
    assert plan(store, prev, force=True)["mode"] == "full"


def test_no_new_sources_is_full(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    assert plan(store, prev)["mode"] == "full"


def test_carried_item_whose_citation_no_longer_validates_is_reanalysed(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    broken = CaseAnalysis.model_validate(prev["analysis"])
    broken.assessments[1].dependencies[0].refs[0].quote = "a quote that is not in the source"
    respond(store, [PEN_KEY])
    p = plan(store, {**prev, "analysis": broken.model_dump()})
    assert p["mode"] == "incremental" and p["reanalyse"] == [PEN_KEY, DPA_KEY]
    assert "provenance" in p["reasons"][DPA_KEY][0]


# -- run behaviour ----------------------------------------------------------


def test_incremental_run_keeps_carried_items_verbatim_and_limits_material_changes(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    resp = respond(store, [PEN_KEY])
    client = FakeClient(lambda **_: rerun_output(store, resp))
    run = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF, client=client)["run"]
    assert run["status"] == "success" and run["analysis_mode"] == "incremental"
    assert run["reanalysed_keys"] == [PEN_KEY] and run["carried_keys"] == [DPA_KEY, RET_KEY]

    user = client.calls[0]["payload"]["user"]
    assert "INCREMENTAL UPDATE" in user and PEN_KEY in user.split("Carried forward")[0]
    assert resp.source_id in user and "Closure evidence missing" in user  # previous affected item is shown

    before = CaseAnalysis.model_validate(prev["analysis"])
    after = CaseAnalysis.model_validate(run["analysis"])
    prev_by_key = {a.requirement_key: a for a in before.assessments}
    for a in after.assessments:
        if a.requirement_key != PEN_KEY:
            assert a == prev_by_key[a.requirement_key]  # byte-for-byte carry-forward
    assert [a.requirement_key for a in after.assessments] == [PEN_KEY, DPA_KEY, RET_KEY]
    assert any("discarded" in w and DPA_KEY in w for w in after.warnings)

    diff = diff_analyses(before, after)
    assert set(diff["material"]) == {PEN_KEY}
    assert set(diff["changed"]) == {PEN_KEY}


def test_new_requirement_from_new_source_is_added(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    resp = respond(store, [PEN_KEY])
    data = rerun_output(store, resp, reword_carried=False)
    data["assessments"].append(assessment(ACME, "Q-009", title="Retest scheduling"))
    run = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF,
                            client=FakeClient([data]))["run"]
    after = CaseAnalysis.model_validate(run["analysis"])
    assert [a.requirement_key for a in after.assessments][-1] == f"{ACME}:Q-009"
    assert diff_analyses(CaseAnalysis.model_validate(prev["analysis"]), after)["added"] == [f"{ACME}:Q-009"]


def test_missing_affected_item_fails_and_keeps_previous_run(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    resp = respond(store, [PEN_KEY])
    client = FakeClient([rerun_output(store, resp, include_pen=False)])
    run = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF, client=client)["run"]
    assert run["status"] == "failed" and PEN_KEY in run["error"]
    assert store.latest_run(ACME)["run_id"] == prev["run_id"]


def test_force_regenerates_the_whole_case(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    respond(store, [PEN_KEY])
    full = initial(store.get_sources(ACME))
    client = FakeClient([full])
    run = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF, client=client, force=True)["run"]
    assert run["analysis_mode"] == "full" and "INCREMENTAL UPDATE" not in client.calls[0]["payload"]["user"]


def test_review_notes_on_carried_items_do_not_need_recheck(cases_copy, tmp_path):
    store, prev = seeded(cases_copy, tmp_path)
    store.save_review(ACME, PEN_KEY, {"review_note": "checked pen-test wording"})
    store.save_review(ACME, DPA_KEY, {"review_note": "checked DPA wording"})
    resp = respond(store, [PEN_KEY])
    run = run_case_analysis(store=store, case_id=ACME, as_of_date=AS_OF,
                            client=FakeClient([rerun_output(store, resp)]))["run"]
    n = mark_reviews_for_recheck(store, ACME, run["run_id"], CaseAnalysis.model_validate(run["analysis"]),
                                 unchanged_keys=set(run["carried_keys"]))
    reviews = {r["requirement_key"]: r for r in store.get_reviews(ACME)}
    assert n == 1 and reviews[PEN_KEY]["needs_recheck"] and "needs_recheck" not in reviews[DPA_KEY]

