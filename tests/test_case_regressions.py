"""Case regressions on the ORIGINAL case sources and RECORDED live model outputs.

The recorded fixtures were produced once by anthropic/claude-sonnet-5.5 via OpenRouter and are
not regenerated here. These assertions pin observed behaviour for regression purposes; they are
not a semantic benchmark and do not prove that a fresh generation would behave the same.
"""

import json
import re
from pathlib import Path

import pytest

from procurement.analysis import validate_analysis
from procurement.drafting import check_update
from procurement.ingest import import_case, questionnaire_rows
from procurement.models import CaseAnalysis, CustomerUpdate

from conftest import ACME, NORTHSTAR, REPO_ROOT

RECORDED = Path(__file__).parent / "fixtures" / "recorded"


def load(name):
    data = json.loads((RECORDED / name).read_text())
    assert data["label"].startswith("RECORDED LIVE MODEL OUTPUT")
    return data


def by_q(analysis: CaseAnalysis, qid: str):
    return next(a for a in analysis.assessments if qid in a.question_ids)


def text_of(a) -> str:
    parts = [p.text for p in a.current_position] + a.gaps + [d.description for d in a.dependencies]
    parts += [c.text for c in a.commitments] + [a.next_action]
    return " ".join(parts).lower()


# -- original inputs --------------------------------------------------------


def test_both_cases_share_question_wording_but_not_statuses():
    acme = questionnaire_rows(import_case(REPO_ROOT, ACME))
    north = questionnaire_rows(import_case(REPO_ROOT, NORTHSTAR))
    assert [r["question"] for r in acme] == [r["question"] for r in north]
    assert [r["status"] for r in acme] != [r["status"] for r in north]
    templated = [r["id"] for r in acme if re.match(r"Control \d+: provide the customer-specific answer", r["question"])]
    assert templated == [f"Q-{n:03d}" for n in range(24, 61)]


# -- recorded live analyses -------------------------------------------------


@pytest.fixture(scope="module")
def acme():
    return CaseAnalysis.model_validate(load("acme_initial_analysis.json")["analysis"])


@pytest.fixture(scope="module")
def north():
    return CaseAnalysis.model_validate(load("northstar_initial_analysis.json")["analysis"])


def test_recorded_outputs_still_pass_provenance_against_original_sources(acme, north):
    validate_analysis(acme, import_case(REPO_ROOT, ACME))
    validate_analysis(north, import_case(REPO_ROOT, NORTHSTAR))


@pytest.mark.parametrize("which", ["acme", "north"])
def test_control_rows_are_one_clarification_and_coverage_is_complete(which, request):
    analysis = request.getfixturevalue(which)
    clar = [a for a in analysis.assessments if a.origin == "clarification"]
    assert len(clar) == 1 and set(clar[0].question_ids) == {f"Q-{n:03d}" for n in range(24, 61)}
    assert clar[0].evidence_state == "missing"
    covered = {q for a in analysis.assessments for q in a.question_ids}
    assert covered | set(analysis.unassessed_question_ids) == {f"Q-{n:03d}" for n in range(1, 61)}


@pytest.mark.parametrize("which", ["acme", "north"])
def test_shared_evidence_notes_are_not_upgraded(which, request):
    analysis = request.getfixturevalue(which)
    soc2 = by_q(analysis, "Q-007")
    assert soc2.evidence_state != "found"
    assert not re.search(r"2026[^.]*\bfinal\b", " ".join(p.text for p in soc2.current_position).replace("not yet final", ""))
    pen = by_q(analysis, "Q-008")
    assert pen.evidence_state != "found"
    assert not any(re.search(r"\b(verified|closed)\b", p.text.lower()) for p in pen.current_position)
    ins = by_q(analysis, "Q-021")
    assert "lapse" in text_of(ins) or "coverage" in text_of(ins)


def test_acme_scope_distinctions(acme):
    retention = by_q(acme, "Q-004")
    assert "trace" in " ".join(retention.gaps).lower()
    assert retention.evidence_state != "found"
    meridian = [p for a in acme.assessments for p in a.current_position if "meridian" in p.text.lower()]
    assert meridian and all(p.applicability == "other_customer" for p in meridian)
    location = by_q(acme, "Q-003")
    assert "inference" in " ".join(location.gaps).lower()
    dpa = by_q(acme, "Q-020")
    assert any(d.basis == "explicit" and "subprocessor" in d.description.lower() for d in dpa.dependencies)
    remediation = [p for p in by_q(acme, "Q-008").current_position if "remediat" in p.text.lower()]
    assert remediation and all(p.kind in ("reported", "documented") for p in remediation)


def test_northstar_scope_distinctions(north):
    connectors = by_q(north, "Q-015")
    kinds = {(p.kind, p.applicability) for p in connectors.current_position}
    assert ("proposed", "customer_specific") in kinds and any(app == "generic" for _, app in kinds)
    assert connectors.evidence_state != "conflicting"
    bluepeak = [p for a in north.assessments for p in a.current_position if "bluepeak" in p.text.lower()]
    assert bluepeak and all(p.applicability == "other_customer" for p in bluepeak)
    for a in north.assessments:
        for c in a.commitments:
            assert "bluepeak" not in c.text.lower() and "24h" not in c.text.lower()
            assert "wednesday" not in (c.date_text or "").lower()
    nov = [c for a in north.assessments for c in a.commitments if "nov 16" in c.text.lower()]
    assert nov and all(c.date_iso is None for c in nov)  # no year in the source, so no invented ISO date
    assert any("records management" in a.title.lower() for a in north.assessments)


def test_recorded_acme_draft_preserves_qualifiers():
    data = load("acme_initial_draft.json")
    update = CustomerUpdate.model_validate(data["generated"])
    _, checks, warnings = check_update(update, import_case(REPO_ROOT, ACME), ACME)
    assert all(c["status"] == "ok" for c in checks) and warnings == []
    text = update.text.lower()
    assert "closure evidence is pending" in text
    assert "intended to be off" in text and "disabled by default" not in text
    # "unless approved" (quoting the draft condition) is fine; an approval/verification claim is not.
    assert not re.search(r"\b(is|are|was|were|has been|have been)\s+(approved|certified|compliant|verified|closed)\b",
                         text)
