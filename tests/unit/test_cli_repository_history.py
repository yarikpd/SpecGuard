from specguard.cli.repositories import saved_repositories, save_repository, delete_repository


def test_repository_shortcuts_persist_and_deduplicate(tmp_path, monkeypatch):
    monkeypatch.setenv('SPECGUARD_CONFIG', str(tmp_path / 'config.json'))
    save_repository('https://github.com/owner/repo.git')
    save_repository('owner/other')
    save_repository('OWNER/REPO')
    assert saved_repositories() == ['OWNER/REPO', 'owner/other']
    delete_repository('owner/repo')
    assert saved_repositories() == ['owner/other']
