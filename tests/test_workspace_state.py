"""Workspace state through Streamlit's AppTest with a counting fake client (no provider use)."""

from datetime import date

from streamlit.testing.v1 import AppTest

from procurement.analysis import run_case_analysis
from procurement.ingest import import_case
from procurement.store import Store

from conftest import ACME, NORTHSTAR, REPO_ROOT
from fakes import FakeClient, analysis, assessment, position, src_ref

AS_OF = date(2026, 10, 19)


def _app(db_path: str, client):
    from pathlib import Path

    from procurement.store import Store
    from procurement.ui import render_workspace

    render_workspace(Store(Path(db_path)), client)


def seeded(tmp_path):
    db = tmp_path / "w.sqlite"
    store = Store(db)
    for cid, title in ((ACME, "Acme pen test"), (NORTHSTAR, "Northstar connectors")):
        sources = import_case(REPO_ROOT, cid)
        store.save_sources(cid, sources)
        a = analysis(cid, AS_OF.isoformat(), [assessment(
            cid, "Q-008", title=title,
            positions=[position("Two Medium findings were open.",
                                [src_ref(sources, "05_evidence/pen_test_summary.txt", 2, 2,
                                         "Two Medium findings were OPEN")], kind="documented")],
            gaps=["Closure evidence missing"], next_action=f"Ask for {title} evidence")])
        run_case_analysis(store=store, case_id=cid, as_of_date=AS_OF, client=FakeClient([a]))
        store.save_review(cid, f"{cid}:Q-008", {"review_note": f"note for {cid}", "owner_override": "Sam"})
    return db, store


def run_app(db, client, case_index=0):
    at = AppTest.from_function(_app, args=(str(db), client), default_timeout=30)
    at.run()
    if case_index:
        at.sidebar.selectbox[0].select_index(case_index).run()
    at.sidebar.date_input[0].set_value(AS_OF).run()
    return at


def all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [e.label for e in at.expander] + [t.value for t in at.text_area]
    parts += [c.value for c in at.code] + [str(t.value) for t in at.text]
    return "\n".join(parts)


def test_reopen_shows_saved_run_without_model_calls(tmp_path):
    db, _ = seeded(tmp_path)
    client = FakeClient([])
    at = run_app(db, client)
    assert not at.exception
    text = all_text(at)
    assert "Acme pen test" in text and "Northstar connectors" not in text
    assert "FAKE/test client" in text
    # Rerenders / view switches make no requests.
    at.run()
    at.sidebar.checkbox[0].check().run()
    assert client.calls == []


def test_case_switch_isolates_assessments_overlays_and_packs(tmp_path):
    db, _ = seeded(tmp_path)
    at = run_app(db, FakeClient([]), case_index=1)
    text = all_text(at)
    assert "Northstar connectors" in text and "Acme pen test" not in text
    assert ACME not in text
    assert "Requests for Sam - case case_2_northstar_telecom" in text  # local override, current case only


def test_original_questionnaire_status_visible_independently(tmp_path):
    db, _ = seeded(tmp_path)
    at = run_app(db, FakeClient([]))
    frames = [df.value for df in at.dataframe]
    q = next(f for f in frames if "original status" in f.columns)
    assert len(q) == 60 and q.iloc[0]["original status"] == "UNKNOWN"


def test_date_change_marks_run_outdated_and_missing_key_disables_actions(tmp_path):
    db, _ = seeded(tmp_path)
    at = AppTest.from_function(_app, args=(str(db), None), default_timeout=30)
    at.run()
    at.sidebar.date_input[0].set_value(date(2026, 9, 1)).run()
    assert any("OUTDATED" in w.value for w in at.warning)
    run_button = next(b for b in at.button if b.label == "Run analysis")
    assert run_button.disabled
