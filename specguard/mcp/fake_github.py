"""Имитация GitHub MCP Server для автономной работы и тестов (Модуль 3).

``FakeGitHub`` хранит небольшую модель репозиториев (коммиты, ветки, PR), а
``FakeMCPTransport`` отвечает на вызовы инструментов так же, как
github-mcp-server v2.0.3: текстовый JSON структур Minimal*, пропуск пустых
полей (omitempty), постраничная выдача файлов, отсутствие родителей у
коммита и тексты ошибок GitHub API.

Управляемые отказы (TEST-006, AC-015) задаются через ``fail()`` и
``on_call()``. Это имитация: она проверяет реакцию адаптера, но не
подтверждает фактическое поведение GitHub.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

from specguard.mcp.transport import (
    SOURCE_MOCK,
    MCPTimeoutError,
    MCPTransportError,
    ToolCallResult,
)

API = "https://api.github.com"
WEB = "https://github.com"

# Готовые виды отказов для fail(error=...). Любая другая строка возвращается
# как текст ошибки инструмента без изменений.
FAULTS = (
    "timeout",       # транспорт: сервер не ответил
    "disconnect",    # транспорт: соединение разорвано
    "rate_limit",    # GitHub: превышен лимит запросов
    "server_error",  # GitHub: 502
    "not_found",     # GitHub: 404
    "forbidden",     # GitHub: 403
    "unauthorized",  # GitHub: 401
    "bad_json",      # ответ без ошибки, но не JSON
)


@dataclass
class FakeFile:
    """Изменённый файл. Счётчики строк по умолчанию считаются по patch."""

    filename: str
    status: str = "modified"
    patch: Optional[str] = None
    previous_filename: Optional[str] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None

    def line_counts(self) -> Tuple[int, int]:
        if self.additions is not None or self.deletions is not None:
            return self.additions or 0, self.deletions or 0
        adds = dels = 0
        for line in (self.patch or "").splitlines():
            if line.startswith("+"):
                adds += 1
            elif line.startswith("-"):
                dels += 1
        return adds, dels


@dataclass
class FakeCommit:
    sha: str
    parents: List[str] = field(default_factory=list)
    files: List[FakeFile] = field(default_factory=list)
    message: str = ""


@dataclass
class FakePullRequest:
    number: int
    head_sha: str
    base_sha: str
    files: List[FakeFile] = field(default_factory=list)
    head_ref: str = "feature"
    base_ref: str = "main"
    fork_repo: Optional[str] = None   # репозиторий-источник, если PR из fork
    fork_deleted: bool = False        # fork удалён: GitHub не отдаёт head.repo
    changed_files: Optional[int] = None  # переопределить заявленное число файлов
    title: str = ""
    state: str = "open"
    merged: bool = False


@dataclass
class _FakeRepo:
    full_name: str
    accessible: bool = True
    commits: Dict[str, FakeCommit] = field(default_factory=dict)
    branches: Dict[str, str] = field(default_factory=dict)
    pulls: Dict[int, FakePullRequest] = field(default_factory=dict)


@dataclass
class _Rule:
    tool: str
    method: Optional[str]
    page: Optional[int]
    sha: Optional[str]
    times: Optional[int]
    error: Optional[str] = None
    callback: Optional[Callable[["FakeGitHub", str, Dict[str, Any]], None]] = None

    def matches(self, tool: str, args: Mapping[str, Any]) -> bool:
        if self.times is not None and self.times <= 0:
            return False
        if tool != self.tool:
            return False
        if self.method is not None and args.get("method") != self.method:
            return False
        if self.page is not None and int(args.get("page") or 1) != self.page:
            return False
        if self.sha is not None and args.get("sha") != self.sha:
            return False
        return True

    def consume(self) -> None:
        if self.times is not None:
            self.times -= 1


def fake_sha(*parts: Any) -> str:
    """Детерминированный 40-символьный SHA для имитации."""
    return hashlib.sha1(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


class FakeGitHub:
    """Модель GitHub для имитации ответов GitHub MCP Server.

    ``auto_provision=True`` — демонстрационный режим: любые репозиторий, PR и
    коммит «существуют» и содержат ``default_files``; ссылка на коммит,
    начинающаяся с ``merge_``, имитирует merge-коммит с двумя родителями.
    """

    def __init__(
        self,
        *,
        supports_parent_syntax: bool = True,
        ignores_parent_suffix: bool = False,
        max_listed_files: int = 3000,
        auto_provision: bool = False,
        default_files: Sequence[FakeFile] = (),
    ):
        # Как GitHub поведёт себя на «<sha>^N», неизвестно до живой проверки:
        # поддержит, отклонит (supports_parent_syntax=False) или отбросит
        # суффикс и вернёт сам коммит (ignores_parent_suffix=True).
        self.supports_parent_syntax = supports_parent_syntax
        self.ignores_parent_suffix = ignores_parent_suffix
        self.max_listed_files = max_listed_files
        self.auto_provision = auto_provision
        self.default_files = list(default_files)
        self.repos: Dict[str, _FakeRepo] = {}
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._rules: List[_Rule] = []

    # ------------------------------------------------------------------ данные

    def add_repo(self, full_name: str, *, accessible: bool = True) -> _FakeRepo:
        repo = _FakeRepo(full_name=full_name, accessible=accessible)
        self.repos[full_name.lower()] = repo
        return repo

    def add_commit(
        self,
        repo: str,
        sha: str,
        *,
        parents: Sequence[str] = (),
        files: Sequence[FakeFile] = (),
        message: str = "",
        branch: Optional[str] = None,
    ) -> FakeCommit:
        target = self._repo(repo, create=True)
        commit = FakeCommit(sha=sha.lower(), parents=[p.lower() for p in parents], files=list(files), message=message)
        target.commits[commit.sha] = commit
        if branch:
            target.branches[branch] = commit.sha
        return commit

    def add_pull_request(self, repo: str, number: int, **kwargs: Any) -> FakePullRequest:
        target = self._repo(repo, create=True)
        pr = FakePullRequest(number=number, **kwargs)
        target.pulls[number] = pr
        return pr

    def pull_request(self, repo: str, number: int) -> FakePullRequest:
        return self._repo(repo).pulls[number]

    # ----------------------------------------------------------- управляемые отказы

    def fail(
        self,
        tool: str,
        *,
        error: str,
        method: Optional[str] = None,
        page: Optional[int] = None,
        sha: Optional[str] = None,
        times: Optional[int] = 1,
    ) -> None:
        """Следующие ``times`` подходящих вызовов завершатся отказом ``error``.

        ``times=None`` — отказ при каждом подходящем вызове.
        """
        self._rules.append(_Rule(tool, method, page, sha, times, error=error))

    def on_call(
        self,
        tool: str,
        callback: Callable[["FakeGitHub", str, Dict[str, Any]], None],
        *,
        method: Optional[str] = None,
        page: Optional[int] = None,
        sha: Optional[str] = None,
        times: Optional[int] = 1,
    ) -> None:
        """Перед ответом на подходящий вызов выполнить ``callback(github, tool, args)``.

        Нужно, например, чтобы сдвинуть head PR посреди чтения.
        """
        self._rules.append(_Rule(tool, method, page, sha, times, callback=callback))

    def calls_to(self, tool: str, method: Optional[str] = None) -> List[Dict[str, Any]]:
        return [args for name, args in self.calls if name == tool and (method is None or args.get("method") == method)]

    def transport(self, source: str = SOURCE_MOCK) -> "FakeMCPTransport":
        return FakeMCPTransport(self, source=source)

    # ------------------------------------------------------------------ вызовы

    def handle(self, tool: str, arguments: Mapping[str, Any]) -> ToolCallResult:
        args = dict(arguments)
        self.calls.append((tool, args))
        for rule in self._rules:
            if not rule.matches(tool, args):
                continue
            rule.consume()
            if rule.callback is not None:
                rule.callback(self, tool, args)
                continue
            return self._fault(rule.error or "", tool, args)

        if tool == "pull_request_read":
            return self._pull_request_read(args)
        if tool == "get_commit":
            return self._get_commit(args)
        return _error(f"unknown tool: {tool}")

    def _pull_request_read(self, args: Dict[str, Any]) -> ToolCallResult:
        method = args.get("method")
        number = int(args.get("pullNumber") or 0)
        prefix = "failed to get pull request files" if method == "get_files" else "failed to get pull request"
        url = f"{API}/repos/{args.get('owner')}/{args.get('repo')}/pulls/{number}"
        repo = self._repo_for(args)
        pr = repo.pulls.get(number) if repo else None
        if pr is None and repo is not None and self.auto_provision:
            pr = self._provision_pr(repo, number)
        if pr is None:
            return _error(f"{prefix}: GET {url}: 404 Not Found []")

        if method == "get":
            return _ok(self._pr_json(repo, pr))
        if method == "get_files":
            page, per_page, problem = _pagination(args, default_per_page=30, max_per_page=100)
            if problem:
                return _error(problem)
            listed = pr.files[: self.max_listed_files]
            chunk = listed[(page - 1) * per_page: page * per_page]
            return _ok([_file_json(f, with_patch=True, with_previous=True) for f in chunk])
        return _error(f"unknown method: {method}")

    def _get_commit(self, args: Dict[str, Any]) -> ToolCallResult:
        ref = str(args.get("sha") or "")
        url = f"{API}/repos/{args.get('owner')}/{args.get('repo')}/commits/{quote(ref)}"
        repo = self._repo_for(args)
        if repo is None:
            return _error(f"failed to get commit: {ref}: GET {url}: 404 Not Found []")
        commit = self._resolve(repo, ref)
        if commit is None:
            return _error(f"failed to get commit: {ref}: No commit found for SHA: {ref}")

        detail = args.get("detail") or "stats"
        if detail not in ("none", "stats", "full_patch"):
            return _error(f'invalid detail "{detail}": must be one of "none", "stats", "full_patch"')
        page, per_page, problem = _pagination(args, default_per_page=30, max_per_page=None)
        if problem:
            return _error(problem)

        data: Dict[str, Any] = {
            "sha": commit.sha,
            "html_url": f"{WEB}/{repo.full_name}/commit/{commit.sha}",
            "commit": {"message": commit.message},
        }
        if detail != "none":
            # stats GitHub считает по всему коммиту, files приходят постранично.
            adds = dels = 0
            for f in commit.files:
                a, d = f.line_counts()
                adds, dels = adds + a, dels + d
            data["stats"] = _omit_zero({"additions": adds, "deletions": dels, "total": adds + dels})
            listed = commit.files[: self.max_listed_files]
            chunk = listed[(page - 1) * per_page: page * per_page]
            if chunk:
                # MinimalCommitFile в v2.0.3 не содержит previous_filename.
                data["files"] = [
                    _file_json(f, with_patch=detail == "full_patch", with_previous=False) for f in chunk
                ]
        return _ok(data)

    # --------------------------------------------------------------- служебное

    def _fault(self, error: str, tool: str, args: Dict[str, Any]) -> ToolCallResult:
        if tool == "get_commit":
            prefix = f"failed to get commit: {args.get('sha')}"
            url = f"{API}/repos/{args.get('owner')}/{args.get('repo')}/commits/{quote(str(args.get('sha')))}"
        else:
            prefix = "failed to get pull request"
            if args.get("method") == "get_files":
                prefix += " files"
            url = f"{API}/repos/{args.get('owner')}/{args.get('repo')}/pulls/{args.get('pullNumber')}"
        if error == "timeout":
            raise MCPTimeoutError("MCP-сервер не ответил вовремя (имитация)")
        if error == "disconnect":
            raise MCPTransportError("соединение с MCP-сервером разорвано (имитация)")
        if error == "rate_limit":
            return _error(f"{prefix}: GitHub API rate limit exceeded. Retry after 10m0s.")
        if error == "server_error":
            return _error(f"{prefix}: GET {url}: 502 Bad Gateway []")
        if error == "not_found":
            return _error(f"{prefix}: GET {url}: 404 Not Found []")
        if error == "forbidden":
            return _error(f"{prefix}: GET {url}: 403 Resource not accessible by personal access token []")
        if error == "unauthorized":
            return _error(f"{prefix}: GET {url}: 401 Bad credentials []")
        if error == "bad_json":
            return ToolCallResult(is_error=False, text="<html><body>502 Bad Gateway</body></html>")
        return _error(error)

    def _repo(self, full_name: str, create: bool = False) -> _FakeRepo:
        key = full_name.lower()
        if key not in self.repos:
            if not create:
                raise KeyError(full_name)
            self.add_repo(full_name)
        return self.repos[key]

    def _repo_for(self, args: Mapping[str, Any]) -> Optional[_FakeRepo]:
        full_name = f"{args.get('owner')}/{args.get('repo')}"
        repo = self.repos.get(full_name.lower())
        if repo is None and self.auto_provision:
            repo = self.add_repo(full_name)
        if repo is None or not repo.accessible:
            return None
        return repo

    def _resolve(self, repo: _FakeRepo, ref: str) -> Optional[FakeCommit]:
        base, parent_index = ref, 0
        match = re.fullmatch(r"(.+?)\^(\d*)", ref)
        if match:
            if not self.supports_parent_syntax:
                return None
            base, parent_index = match.group(1), int(match.group(2) or 1)
            if self.ignores_parent_suffix:
                parent_index = 0

        commit = self._lookup(repo, base)
        if commit is None and self.auto_provision:
            commit = self._provision_commit(repo, base)
        if commit is None or parent_index == 0:
            return commit
        if parent_index > len(commit.parents):
            return None
        return repo.commits.get(commit.parents[parent_index - 1])

    @staticmethod
    def _lookup(repo: _FakeRepo, ref: str) -> Optional[FakeCommit]:
        lowered = ref.lower()
        if lowered in repo.commits:
            return repo.commits[lowered]
        if ref in repo.branches:
            return repo.commits.get(repo.branches[ref])
        if re.fullmatch(r"[0-9a-f]{4,39}", lowered):
            found = [c for sha, c in repo.commits.items() if sha.startswith(lowered)]
            if len(found) == 1:
                return found[0]
        return None

    def _provision_commit(self, repo: _FakeRepo, ref: str) -> FakeCommit:
        sha = ref.lower() if re.fullmatch(r"[0-9a-f]{40}", ref.lower()) else fake_sha(repo.full_name, ref)
        count = 2 if ref.startswith("merge_") else 1
        parents = [fake_sha(sha, "parent", i) for i in range(1, count + 1)]
        for parent in parents:
            repo.commits.setdefault(parent, FakeCommit(sha=parent))
        commit = FakeCommit(sha=sha, parents=parents, files=list(self.default_files), message=f"Демо-коммит {ref}")
        repo.commits[sha] = commit
        if sha != ref.lower():
            repo.branches[ref] = sha
        return commit

    def _provision_pr(self, repo: _FakeRepo, number: int) -> FakePullRequest:
        pr = FakePullRequest(
            number=number,
            head_sha=fake_sha(repo.full_name, "pr", number, "head"),
            base_sha=fake_sha(repo.full_name, "pr", number, "base"),
            files=list(self.default_files),
            title=f"Демо-PR #{number}",
        )
        repo.pulls[number] = pr
        return pr

    def _pr_json(self, repo: _FakeRepo, pr: FakePullRequest) -> Dict[str, Any]:
        adds = dels = 0
        for f in pr.files:
            a, d = f.line_counts()
            adds, dels = adds + a, dels + d
        head_repo = None if pr.fork_deleted else (pr.fork_repo or repo.full_name)
        data: Dict[str, Any] = {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "draft": False,
            "merged": pr.merged,
            "html_url": f"{WEB}/{repo.full_name}/pull/{pr.number}",
            "head": _branch_json(pr.head_ref, pr.head_sha, head_repo),
            "base": _branch_json(pr.base_ref, pr.base_sha, repo.full_name),
        }
        changed = pr.changed_files if pr.changed_files is not None else len(pr.files)
        data.update(_omit_zero({"additions": adds, "deletions": dels, "changed_files": changed, "commits": 1}))
        return data


class FakeMCPTransport:
    """Транспорт, который вместо процесса GitHub MCP Server спрашивает FakeGitHub."""

    def __init__(self, github: FakeGitHub, source: str = SOURCE_MOCK):
        self.github = github
        self.source = source

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolCallResult:
        return self.github.handle(name, arguments)


def _ok(payload: Any) -> ToolCallResult:
    return ToolCallResult(is_error=False, text=json.dumps(payload, ensure_ascii=False))


def _error(text: str) -> ToolCallResult:
    return ToolCallResult(is_error=True, text=text)


def _omit_zero(values: Dict[str, int]) -> Dict[str, int]:
    return {key: value for key, value in values.items() if value}


def _branch_json(ref: str, sha: str, repo_full_name: Optional[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {"ref": ref, "sha": sha}
    if repo_full_name:
        data["repo"] = {"full_name": repo_full_name}
    return data


def _file_json(f: FakeFile, *, with_patch: bool, with_previous: bool) -> Dict[str, Any]:
    adds, dels = f.line_counts()
    data: Dict[str, Any] = {"filename": f.filename}
    if f.status:
        data["status"] = f.status
    data.update(_omit_zero({"additions": adds, "deletions": dels, "changes": adds + dels}))
    if with_patch and f.patch:
        data["patch"] = f.patch
    if with_previous and f.previous_filename:
        data["previous_filename"] = f.previous_filename
    return data


def _pagination(
    args: Mapping[str, Any], *, default_per_page: int, max_per_page: Optional[int]
) -> Tuple[int, int, Optional[str]]:
    page = int(args.get("page") or 1)
    per_page = int(args.get("perPage") or default_per_page)
    if page < 1 or per_page < 1:
        return page, per_page, "invalid pagination: page and perPage must be >= 1"
    if max_per_page is not None and per_page > max_per_page:
        return page, per_page, f"invalid pagination: perPage must be <= {max_per_page}"
    return page, per_page, None
