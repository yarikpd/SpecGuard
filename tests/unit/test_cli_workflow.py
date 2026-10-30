from typer.testing import CliRunner

from specguard.cli import main
from specguard.cli.history import HistoryStore


def test_demo_audit_history_and_dispute_use_saved_materials(tmp_path, monkeypatch):
    store = HistoryStore(tmp_path / "history")
    monkeypatch.setattr(main, "history_store", store)
    spec = tmp_path / "spec.md"
    spec.write_text("# Requirements\n\nFR-001: Password must contain at least 8 characters.\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main.app, ["audit", "--demo", "--spec", str(spec), "--repo", "owner/repo", "--target", "1"])
    assert result.exit_code == 0, result.output
    report = store.list()[0]
    assert report.saved_git_context is not None and report.saved_requirements
    spec.unlink()
    assert runner.invoke(main.app, ["history"]).exit_code == 0
    assert runner.invoke(main.app, ["show", report.audit_id]).exit_code == 0
    result = runner.invoke(main.app, ["dispute", report.audit_id, "--req", report.findings[0].req_id, "--comment", "Please reconsider", "--demo"])
    assert result.exit_code == 0, result.output
    assert len(store.list_disputes(report.audit_id)) == 1


def test_live_mode_does_not_silently_use_mocks(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "history_store", HistoryStore(tmp_path / "history"))
    monkeypatch.setenv("SPECGUARD_CONFIG", str(tmp_path / "missing.json"))
    for name in ("SPECGUARD_LLM_URL", "SPECGUARD_LLM_MODEL", "SPECGUARD_EMBEDDING_URL", "SPECGUARD_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    spec = tmp_path / "spec.txt"
    spec.write_text("FR-001: test", encoding="utf-8")
    result = CliRunner().invoke(main.app, ["audit", "--spec", str(spec), "--repo", "owner/repo", "--target", "1"])
    assert result.exit_code != 0
    assert not (tmp_path / "history").exists()


def test_configure_one_model_preserves_other_profile(tmp_path, monkeypatch):
    from specguard.cli.config import load_config
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"embedding": {"base_url": "http://embedding/v1", "model": "old-embedding"}}))
    monkeypatch.setenv("SPECGUARD_CONFIG", str(path))
    monkeypatch.setenv("SPECGUARD_EMBEDDING_MODEL", "environment-only")
    choices = iter(["llm", "url", "model", "new-llm", None, None])
    monkeypatch.setattr(main, "_choose", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(main.Prompt, "ask", lambda *args, **kwargs: "http://llm/v1")
    monkeypatch.setattr(main, "discover_models", lambda config: ["new-llm"])
    main._configure()
    saved = json.loads(path.read_text())
    assert saved["llm"] == {"base_url": "http://llm/v1", "model": "new-llm"}
    assert saved["embedding"] == {"base_url": "http://embedding/v1", "model": "old-embedding"}
    assert load_config().llm.model == "new-llm"


def test_embedding_key_is_hidden_and_available_to_clients(tmp_path, monkeypatch):
    from specguard.cli.config import load_config
    path = tmp_path / "config.json"
    monkeypatch.setenv("SPECGUARD_CONFIG", str(path))
    choices = iter(["embedding", "key", None, None])
    monkeypatch.setattr(main, "_choose", lambda *args, **kwargs: next(choices))
    def input_key(*args, **kwargs):
        assert kwargs["hide_input"] is True
        return "secret-embedding-key"
    monkeypatch.setattr(main.typer, "prompt", input_key)
    main._configure()
    assert load_config().embedding.api_key == "secret-embedding-key"
    assert not path.exists() or "secret-embedding-key" not in path.read_text()
    assert not load_config().llm.api_key


def test_multiple_specs_save_independent_snapshots_and_requirements(tmp_path):
    from specguard.cli.config import AppConfig, APIConfig
    from specguard.cli.service import run_audit
    first = tmp_path / "one.md"
    second = tmp_path / "two.md"
    first.write_text("FR-001: Password must contain at least eight characters.", encoding="utf-8")
    second.write_text("FR-001: The application must log failed attempts.", encoding="utf-8")
    store = HistoryStore(tmp_path / "history")
    report = run_audit(AppConfig(APIConfig(), APIConfig()), store, [str(first), str(second)],
                       "owner/repo", "1", "", demo=True)
    assert len(report.metadata["spec_snapshots"]) == 2
    assert {r.req_id for r in report.saved_requirements} >= {"DOC1-FR-001", "DOC2-FR-001"}
    assert {r.source_file for r in report.saved_requirements} >= {str(first), str(second)}
    assert all((store.directory / "snapshots" / (entry["snapshot_id"] + ".json")).exists()
               for entry in report.metadata["spec_snapshots"])


def test_select_multiple_specs(tmp_path, monkeypatch):
    first, second = tmp_path / "one.md", tmp_path / "two.txt"
    first.write_text("first")
    second.write_text("second")
    choices = iter([first, "add", second, "done"])
    monkeypatch.setattr(main, "_choose", lambda *args, **kwargs: next(choices))
    assert main._select_specs(tmp_path) == [str(first), str(second)]


def test_reuse_saved_specs_after_source_deleted_and_add_new_file(tmp_path):
    from specguard.cli.config import AppConfig, APIConfig
    from specguard.cli.service import run_audit
    config = AppConfig(APIConfig(), APIConfig())
    store = HistoryStore(tmp_path / 'history')
    first = tmp_path / 'first.md'
    first.write_text('FR-001: Password must contain at least eight characters.')
    old = run_audit(config, store, str(first), 'owner/repo', '1', '', demo=True)
    first.unlink()
    second = tmp_path / 'second.md'
    second.write_text('FR-002: Log failed login attempts.')
    specs = old.metadata['spec_snapshots'] + [str(second)]
    new = run_audit(config, store, specs, 'owner/repo', '2', '', demo=True)
    assert len(new.metadata['spec_snapshots']) == 2
    assert new.metadata['spec_snapshots'][0]['snapshot_id'] == old.metadata['spec_snapshots'][0]['snapshot_id']
    assert {r.req_id for r in new.saved_requirements} >= {'FR-001', 'FR-002'}
    assert len(store.list()) == 2
