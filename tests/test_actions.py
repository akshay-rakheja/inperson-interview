from procurement.actions import UNASSIGNED, build_owner_action_packs, pack_as_text
from procurement.models import CaseAnalysis

from conftest import ACME, NORTHSTAR
from fakes import analysis, assessment, position

REF = {"source_id": f"{ACME}:x.txt@abc", "start_line": 1, "end_line": 1, "quote": "q"}


def make():
    dep = {"description": "DPA blocked until subprocessors confirmed", "basis": "explicit", "refs": [REF]}
    commit = {"text": "Retest planned", "owner": None, "date_iso": None, "date_text": None,
              "date_kind": "unknown", "refs": [REF]}
    return CaseAnalysis.model_validate(analysis(ACME, "2026-10-19", [
        assessment(ACME, "Q-004", title="Retention", positions=[position("a", [REF]), position("b", [REF])],
                   gaps=["Trace retention unknown"], next_action="Confirm trace retention",
                   owner="Anthony (Engineering)", deps=[dep]),
        assessment(ACME, "Q-008", title="Pen test", positions=[position("c", [REF])],
                   gaps=["Closure evidence missing"], next_action="", owner="Anthony (Engineering)",
                   commitments=[commit]),
        assessment(ACME, "Q-020", title="DPA", gaps=["Subprocessors"], next_action="List subprocessors",
                   owner=None),
        assessment(ACME, "Q-001", title="Done", gaps=[], next_action="", owner="Parsa"),
    ]))


def test_groups_by_owner_with_unassigned_last_and_preserves_details():
    packs = build_owner_action_packs(analysis=make())
    assert [p["owner_label"] for p in packs] == ["Anthony (Engineering)", UNASSIGNED]
    anthony = packs[0]["items"]
    assert [i["requirement_key"] for i in anthony] == [f"{ACME}:Q-004", f"{ACME}:Q-008"]
    assert anthony[0]["refs"] == [REF]  # deduplicated across positions and dependency
    assert anthony[0]["dependencies"][0]["basis"] == "explicit"
    assert anthony[1]["ask"] == "Clarify missing information for Pen test"
    assert anthony[1]["commitments"][0]["date_iso"] is None
    assert all(i["requirement_key"] != f"{ACME}:Q-001" for p in packs for i in p["items"])


def test_local_override_wins_and_foreign_or_removed_keys_are_ignored():
    overrides = {f"{ACME}:Q-020": "  Mina (Legal) ", f"{NORTHSTAR}:Q-004": "Sam", f"{ACME}:Q-999": "Sam"}
    packs = build_owner_action_packs(analysis=make(), owner_overrides=overrides)
    labels = [p["owner_label"] for p in packs]
    assert labels == ["Anthony (Engineering)", "Mina (Legal)"]
    mina = packs[1]["items"][0]
    assert mina["owner_basis"] == "local_override" and mina["suggested_owner"] is None


def test_text_export_keeps_unconfirmed_dates_and_disclaimer():
    pack = build_owner_action_packs(analysis=make())[0]
    text = pack_as_text(pack, case_id=ACME, run_id="run-1", outdated=True)
    assert "date unconfirmed" in text and "OUTDATED" in text and "not an accepted assignment" in text
