"""Locally saved repository shortcuts, independent of audit history."""
import json

from specguard.cli.config import config_path
from specguard.mcp.github_client import parse_repo


def saved_repositories():
    path = config_path().parent / 'repositories.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []


def _write(repos):
    path = config_path().parent / 'repositories.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(repos, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_repository(value):
    owner, repo = parse_repo(value)
    name = f'{owner}/{repo}'
    repos = [item for item in saved_repositories() if item.casefold() != name.casefold()]
    _write([name, *repos])
    return name


def delete_repository(value):
    _write([item for item in saved_repositories() if item.casefold() != value.casefold()])
