import os
import subprocess
import sys
from pathlib import Path

from procurement import config
from procurement.ingest import discover_cases, import_case, questionnaire_rows, scan_case
from procurement.store import Store

from conftest import ACME, NORTHSTAR


def test_discovers_both_cases_and_catalogs_14_files(cases_copy):
    cases = {c["case_id"] for c in discover_cases(cases_copy)}
    assert cases == {ACME, NORTHSTAR}
    for case_id in cases:
        sources, warnings = scan_case(cases_copy, case_id)
        assert len(sources) == 14, [s.relative_path for s in sources]
        assert warnings == []  # .DS_Store is hidden, so silently skipped


def test_questionnaire_keeps_60_rows_and_original_statuses(cases_copy):
    acme = questionnaire_rows(import_case(cases_copy, ACME))
    north = questionnaire_rows(import_case(cases_copy, NORTHSTAR))
    assert len(acme) == len(north) == 60
    assert acme[0]["id"] == "Q-001" and acme[0]["status"] == "UNKNOWN" and acme[0]["_line"] == 2
    assert north[0]["status"] == "IN_PROGRESS"
    assert acme[3]["draft_answer"] == "30 days?"


def test_files_without_trailing_newline_are_anchored(cases_copy):
    sources = {s.relative_path: s for s in import_case(cases_copy, ACME)}
    msgs = sources["03_internal/internal_messages.txt"]
    assert msgs.metadata["trailing_newline"] is False
    assert len(msgs.lines()) == 8
    assert msgs.lines()[-1].startswith("Legal: please don't guess")


def test_unsupported_file_warns(cases_copy):
    (cases_copy / "CASE_1_Acme_Bank" / "05_evidence" / "scan.pdf").write_bytes(b"%PDF-1.4")
    sources, warnings = scan_case(cases_copy, ACME)
    assert len(sources) == 14
    assert any("scan.pdf" in w for w in warnings)


def test_import_is_idempotent_and_survives_reopen(cases_copy, tmp_path):
    db = tmp_path / "w.sqlite"
    store = Store(db)
    store.save_sources(ACME, import_case(cases_copy, ACME))
    store.save_sources(ACME, import_case(cases_copy, ACME))
    assert len(Store(db).get_source_versions(ACME)) == 14


def test_same_content_is_case_isolated(cases_copy, tmp_path):
    store = Store(tmp_path / "w.sqlite")
    store.save_sources(ACME, import_case(cases_copy, ACME))
    store.save_sources(NORTHSTAR, import_case(cases_copy, NORTHSTAR))
    a = {s.relative_path: s for s in store.get_sources(ACME)}
    n = {s.relative_path: s for s in store.get_sources(NORTHSTAR)}
    soc2 = "05_evidence/soc2_status.txt"
    assert a[soc2].content_sha256 == n[soc2].content_sha256
    assert a[soc2].source_id != n[soc2].source_id
    assert all(s.case_id == ACME for s in a.values())


def test_changed_content_creates_retained_version(cases_copy, tmp_path):
    store = Store(tmp_path / "w.sqlite")
    store.save_sources(ACME, import_case(cases_copy, ACME))
    path = cases_copy / "CASE_1_Acme_Bank" / "05_evidence" / "soc2_status.txt"
    path.write_text(path.read_text() + "\nUpdated line.")
    store.save_sources(ACME, import_case(cases_copy, ACME))
    current = [s for s in store.get_sources(ACME) if s.relative_path.endswith("soc2_status.txt")]
    versions = [s for s in store.get_source_versions(ACME) if s.relative_path.endswith("soc2_status.txt")]
    assert len(current) == 1 and "Updated line." in current[0].text
    assert len(versions) == 2


def test_runs_reviews_drafts_survive_reopen_and_failures_do_not_replace(tmp_path):
    db = tmp_path / "w.sqlite"
    store = Store(db)
    ok = store.save_run(ACME, {"status": "success", "input_hash": "h1", "analysis": {"x": 1}, "error": None})
    store.save_run(ACME, {"status": "failed", "input_hash": "h2", "analysis": None, "error": "timeout"})
    store.save_review(ACME, f"{ACME}:Q-008", {"review_note": "checked"})
    store.save_review(ACME, f"{ACME}:Q-008", {"owner_override": "Anthony"})
    draft = store.save_draft(ACME, ok, {"generated": {"text": "hi"}})
    store.invalidate_drafts(ACME, "new source")

    reopened = Store(db)
    assert reopened.latest_run(ACME)["run_id"] == ok
    assert [r["status"] for r in reopened.list_runs(ACME)] == ["failed", "success"]
    review = reopened.get_reviews(ACME)[0]
    assert review["review_note"] == "checked" and review["owner_override"] == "Anthony"
    assert len(review["history"]) == 1
    d = reopened.get_drafts(ACME)[0]
    assert d["draft_id"] == draft and d["stale"] and d["stale_reason"] == "new source"
    assert reopened.get_drafts(NORTHSTAR) == []


def test_review_key_must_be_case_scoped(tmp_path):
    store = Store(tmp_path / "w.sqlite")
    try:
        store.save_review(ACME, f"{NORTHSTAR}:Q-001", {"review_note": "x"})
    except ValueError:
        return
    raise AssertionError("cross-case review key accepted")


# -- configuration -----------------------------------------------------------


def test_env_loading_respects_process_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=dummy-from-file\nPROCUREMENT_MODEL=file/model\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("PROCUREMENT_MODEL", "process/model")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    try:
        config.load_env(env)
        settings = config.get_settings()
    finally:
        os.environ.pop("OPENROUTER_API_KEY", None)
    assert settings.has_key is True
    assert settings.model == "process/model"
    assert settings.base_url == "https://openrouter.ai/api/v1"


def test_blank_key_is_a_visible_state(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    settings = config.get_settings()
    assert settings.has_key is False
    assert "OPENROUTER_API_KEY=" in settings.status_message


def test_ensure_env_file_preserves_existing(tmp_path):
    env = tmp_path / ".env"
    assert config.ensure_env_file(env) is True
    assert "PROCUREMENT_MODEL=anthropic/claude-sonnet-5.5" in env.read_text()
    env.write_text("OPENROUTER_API_KEY=dummy\n")
    assert config.ensure_env_file(env) is False
    assert env.read_text() == "OPENROUTER_API_KEY=dummy\n"


def test_env_example_has_no_key(repo_root):
    text = (repo_root / ".env.example").read_text()
    assert "OPENROUTER_API_KEY=\n" in text
    assert "PROCUREMENT_MODEL=anthropic/claude-sonnet-5.5" in text


def test_runtime_files_are_git_ignored(repo_root):
    for path in [".env", "data/workspace.sqlite"]:
        result = subprocess.run(["git", "check-ignore", path], cwd=repo_root, capture_output=True)
        assert result.returncode == 0, path
    assert subprocess.run(["git", "check-ignore", ".env.example"], cwd=repo_root).returncode == 1
