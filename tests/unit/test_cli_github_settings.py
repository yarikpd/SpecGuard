import json
import stat

from specguard.cli.config import load_github_token, save_api_key


def test_github_token_survives_reload_and_preserves_model_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('SPECGUARD_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.delenv('GITHUB_PERSONAL_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    save_api_key('llm', 'model-secret')
    save_api_key('github', 'github-secret')
    assert load_github_token() == 'github-secret'
    assert json.loads((tmp_path / 'secrets.json').read_text())['llm'] == 'model-secret'
    assert stat.S_IMODE((tmp_path / 'secrets.json').stat().st_mode) == 0o600
    monkeypatch.setenv('GITHUB_TOKEN', 'env-secret')
    assert load_github_token() == 'env-secret'
    monkeypatch.delenv('GITHUB_TOKEN')
    save_api_key('github', '')
    assert load_github_token() == ''


def test_catalog_uses_saved_token(tmp_path, monkeypatch):
    from specguard.cli import catalog
    monkeypatch.setenv('SPECGUARD_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.delenv('GITHUB_PERSONAL_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    save_api_key('github', 'saved-secret')
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(catalog, 'create_stdio_transport', create)
    catalog.GitHubCatalog()
    assert captured['token'] == 'saved-secret'


def test_server_picker_navigates_and_rejects_non_executable(tmp_path, monkeypatch):
    from specguard.cli import main
    folder = tmp_path / 'tools'
    folder.mkdir()
    invalid = folder / 'readme.txt'
    invalid.write_text('text')
    executable = folder / 'github-mcp-server'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    choices = iter([folder, invalid, executable])
    monkeypatch.setattr(main, '_choose', lambda *args, **kwargs: next(choices))
    assert main._select_mcp_server(tmp_path) == str(executable.resolve())


def test_cancel_server_picker_keeps_current_path(monkeypatch):
    from specguard.cli import main
    monkeypatch.setenv('SPECGUARD_MCP_SERVER', '/existing/server')
    choices = iter(['server', None, None])
    monkeypatch.setattr(main, '_choose', lambda *args, **kwargs: next(choices))
    main._github_settings()
    assert main.os.environ['SPECGUARD_MCP_SERVER'] == '/existing/server'


def test_repository_picker_loads_before_asking_for_search(monkeypatch):
    from specguard.cli import main
    calls = []
    class Catalog:
        def repositories(self, page, query):
            calls.append((page, query))
            return [('owner/repo', 'owner/repo')]
        def close(self): pass
    monkeypatch.setattr(main, 'GitHubCatalog', Catalog)
    monkeypatch.setattr(main, '_choose', lambda *args, **kwargs: 'owner/repo')
    def unexpected_prompt(*args, **kwargs):
        raise AssertionError('Search must not block initial loading')
    monkeypatch.setattr(main.Prompt, 'ask', unexpected_prompt)
    assert main._catalog_pick('repo') == 'owner/repo'
    assert calls == [(1, '')]


def test_mcp_server_path_persists(tmp_path, monkeypatch):
    from specguard.cli.config import load_mcp_server, save_mcp_server
    monkeypatch.setenv('SPECGUARD_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.delenv('SPECGUARD_MCP_SERVER', raising=False)
    save_mcp_server('/tools/github-mcp-server')
    assert load_mcp_server() == '/tools/github-mcp-server'
    save_mcp_server('')
    assert load_mcp_server() == ''
