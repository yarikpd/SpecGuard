"""GitHub MCP Client and Git Adapters (Module 3).

Read-only адаптер к GitHub MCP Server (MCP_INTEGRATION.md): разрешает PR или
коммит, собирает список изменений с patch, фиксирует ревизии и происхождение
данных и явно отражает неполноту, ошибки и несогласованность ревизий.

Состав:
- ``BaseGitClient`` — контракт для оркестратора (Модуль 1), включая
  ``get_context`` (выбор между PR и коммитом).
- ``GitHubMCPClient`` — адаптер: нормализация, пагинация, повторы, граничные
  случаи. Работает поверх ``MCPTransport``: живой ``StdioMCPTransport``
  (запускается сам по переменным окружения) или имитация ``FakeMCPTransport``.
- ``MockGitHubClient`` — тот же адаптер поверх имитации GitHub MCP, для
  реалистичных сценариев с отказами без интернета и токенов.
- ``MockGitClient`` — простая заглушка с фиксированными ответами и
  «волшебными» ссылками (merge_, no_base, missing_parent); на ней построены
  тесты оркестратора, её поведение менять нельзя.

Сверено с исходниками github/github-mcp-server v2.0.3:
- ``pull_request_read(get)`` → MinimalPullRequest: head/base с sha и
  repo.full_name; ``changed_files`` опускается, если равно 0;
- ``pull_request_read(get_files)`` → MinimalPRFile с previous_filename,
  perPage ≤ 100, GitHub отдаёт не более 3000 файлов;
- ``get_commit`` → MinimalCommit БЕЗ родителей и без previous_filename;
  файлы постранично (perPage по умолчанию 30), stats — по всему коммиту.

Требует живой проверки (AC-014): разрешает ли GitHub ссылку ``<sha>^N``
через get_commit и строится ли список файлов merge-коммита относительно
первого родителя.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from specguard.mcp.context import GitHubContext
from specguard.mcp.errors import GitSourceError, GitSourceErrorKind, classify_tool_error
from specguard.mcp.fake_github import FakeFile, FakeGitHub
from specguard.mcp.stdio_transport import create_stdio_transport
from specguard.mcp.transport import (
    SOURCE_MOCK,
    MCPCompatibilityError,
    MCPConfigurationError,
    MCPTimeoutError,
    MCPTransport,
    MCPTransportError,
    ToolCallResult,
)
from specguard.models.schema import FileChange, GitContext

# Адаптер вызывает только эти инструменты и методы (NFR-002: права задаёт
# приложение, а не данные репозитория).
ALLOWED_TOOLS: Dict[str, Optional[Set[str]]] = {
    "pull_request_read": {"get", "get_files"},
    "get_commit": None,
}

PER_PAGE = 100
MAX_FILE_PAGES = 30  # 30 × 100 = 3000 — предел GitHub для списков файлов
MAX_PARENTS = 16

MOCK_NOTE = (
    "Данные получены из имитации GitHub (мок-режим), а не из живого GitHub MCP; "
    "они не подтверждают реальное состояние репозитория."
)
EMPTY_FILES_NOTE = (
    "Источник вернул пустой список изменённых файлов; это не подтверждает отсутствие изменений."
)
PR_BASE_NOTE = (
    "Diff PR строится GitHub относительно общей базы веток (merge-base); base_sha — "
    "вершина базовой ветки и может с ней не совпадать."
)
MERGE_UNVERIFIED_NOTE = (
    "Список файлов merge-коммита сформирован GitHub; то, что он построен относительно "
    "первого родителя, ещё не подтверждено живой проверкой (AC-014)."
)
PARENTS_UNKNOWN_NOTE = (
    "Базовая ревизия неизвестна: SHA родителя не зафиксирован, а если это merge-коммит, "
    "сравнение с первым родителем не подтверждено."
)

_SHA_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_REF_RE = re.compile(r"[A-Za-z0-9._/\-]{1,255}")
_REPO_RE = re.compile(
    r"(?:(?:https?://)?(?:www\.)?github\.com/)?"
    r"([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([A-Za-z0-9._-]{1,100}?)(?:\.git)?/?"
)


def parse_repo(repo: str) -> Tuple[str, str]:
    """``owner/repo`` или ссылка на GitHub → (owner, repo)."""
    match = _REPO_RE.fullmatch(repo.strip()) if isinstance(repo, str) else None
    if not match or match.group(2) in (".", ".."):
        raise ValueError(f"Ожидается репозиторий в формате owner/repo, получено: {repo!r}")
    return match.group(1), match.group(2)


class BaseGitClient(ABC):
    """Абстрактный интерфейс для получения изменений из Git/GitHub."""

    @abstractmethod
    def get_pr_context(self, repo: str, pr_number: int) -> GitContext:
        """Получить контекст PR (файлы, diff, base/head SHA)."""

    @abstractmethod
    def get_commit_context(self, repo: str, commit_sha: str) -> GitContext:
        """Получить контекст коммита. Для merge-коммита сравнить с первым родителем."""

    def get_context(self, repo: str, ref: str) -> GitContext:
        """Получить контекст изменений для PR или коммита.

        Определяет тип ref:
        - Если ref указывает на PR (число, 'PR#123', 'pull/123/head', 'refs/pull/123/merge'),
          вызывает get_pr_context.
        - Иначе считает ref хэшем коммита или веткой и вызывает get_commit_context.
        """
        clean_ref = str(ref).strip()
        pr_match = re.match(
            r"^(?:(?:refs/)?pull/|pr\s*#?|#)?(\d+)(?:/head|/merge)?$",
            clean_ref,
            re.IGNORECASE,
        )
        if pr_match and (
            (clean_ref.isdigit() and len(clean_ref) <= 7)
            or clean_ref.lower().startswith(("pr", "#", "pull/", "refs/pull/"))
        ):
            pr_num = int(pr_match.group(1))
            return self.get_pr_context(repo, pr_num)
        return self.get_commit_context(repo, clean_ref)


class GitHubMCPClient(BaseGitClient):
    """Адаптер к GitHub MCP Server.

    Бросает ``GitSourceError``, если без ответа анализ невозможен (нет
    метаданных PR, коммит не найден, нет доступа). Частичные сбои отражаются в
    ``GitContext.limitations`` и ``is_partial``, а повторы — в ``diagnostics``.

    Аргументы:
        mcp_server_cmd: путь к ``github-mcp-server`` (или команда списком);
            по умолчанию ``SPECGUARD_MCP_SERVER`` или поиск в PATH.
        auth_token: токен GitHub; по умолчанию ``GITHUB_PERSONAL_ACCESS_TOKEN``
            или ``GITHUB_TOKEN``. Передаётся серверу только через окружение.
        transport: готовый транспорт. Если не передан, при первом вызове
            запускается сервер по stdio; без сервера или токена —
            ``NOT_CONFIGURED`` (подмены данными мока нет).
        call_timeout: таймаут одного вызова живого сервера, секунды.
        retry_delays: паузы перед повторами при таймауте и недоступности;
            число элементов = число повторов.
        consistency_retries: сколько раз перечитать файлы PR, если head
            изменился во время чтения.
        resolve_parents: определять родителей коммита запросами ``<sha>^N``.
        merge_diff_semantics_verified: включить после живой проверки того,
            что GitHub строит список файлов merge-коммита от первого родителя.
    """

    def __init__(
        self,
        mcp_server_cmd: Optional[Union[str, Sequence[str]]] = None,
        auth_token: Optional[str] = None,
        *,
        transport: Optional[MCPTransport] = None,
        call_timeout: Optional[float] = None,
        retry_delays: Sequence[float] = (1.0, 3.0),
        consistency_retries: int = 1,
        resolve_parents: bool = True,
        merge_diff_semantics_verified: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ):
        # Токен хранится только здесь и не попадает в GitContext.
        self.mcp_server_cmd = mcp_server_cmd
        self._auth_token = auth_token
        self._transport = transport
        self.call_timeout = call_timeout
        self.retry_delays = tuple(retry_delays)
        self.consistency_retries = max(0, consistency_retries)
        self.resolve_parents = resolve_parents
        self.merge_diff_semantics_verified = merge_diff_semantics_verified
        self._sleep = sleep

    # ------------------------------------------------------------------ PR

    def get_pr_context(self, repo: str, pr_number: int) -> GitHubContext:
        owner, name = parse_repo(repo)
        if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
            raise ValueError(f"Номер PR должен быть положительным целым, получено: {pr_number!r}")

        session = _ReadSession(self)
        pr_args = {"owner": owner, "repo": name, "pullNumber": pr_number}
        label = f"PR #{pr_number}"

        pr = session.call_dict("pull_request_read", {"method": "get", **pr_args}, what=f"{label}: метаданные")
        head_sha, base_sha = _pr_sha(pr, "head", label), _pr_sha(pr, "base", label)

        limitations: List[str] = []
        consistent: Optional[bool] = None
        files: List[FileChange] = []
        file_notes: List[str] = []
        complete = True

        for attempt in range(self.consistency_retries + 1):
            files, file_notes, complete = self._read_pr_files(
                session, pr_args, label, expected=_optional_int(pr.get("changed_files"))
            )
            # Повторное чтение SHA обнаруживает часть изменений, но не даёт атомарности.
            try:
                pr_after = session.call_dict(
                    "pull_request_read", {"method": "get", **pr_args}, what=f"{label}: повторная проверка SHA"
                )
                head_after = _pr_sha(pr_after, "head", label)
                base_after = _pr_sha(pr_after, "base", label)
            except GitSourceError as exc:
                consistent = None
                limitations.append(
                    f"Не удалось повторно проверить SHA PR после чтения ({exc}); "
                    "согласованность ревизий не подтверждена."
                )
                break

            if head_after == head_sha:
                consistent = True
                if base_after != base_sha:
                    session.note(
                        f"Вершина базовой ветки сдвинулась во время чтения: {_short(base_sha)} → {_short(base_after)}."
                    )
                pr, base_sha = pr_after, base_after
                break

            consistent = False
            session.note(f"Head PR изменился во время чтения: {_short(head_sha)} → {_short(head_after)}.")
            pr, head_sha, base_sha = pr_after, head_after, base_after
            if attempt < self.consistency_retries:
                session.note("Список файлов PR перечитан.")

        if consistent is False:
            limitations.append(
                f"Head PR менялся во время чтения (последний SHA {head_sha}); "
                "список файлов может относиться к другой ревизии."
            )
        limitations.extend(file_notes)

        base_repo = _branch_repo(pr.get("base")) or f"{owner}/{name}"
        head_repo = _branch_repo(pr.get("head"))
        if head_repo is None:
            limitations.append(
                "Репозиторий-источник PR недоступен (fork удалён или скрыт); "
                "чтение кода по head SHA может не удаться."
            )
        elif head_repo.lower() != base_repo.lower():
            session.note(f"PR из fork: {head_repo} → {base_repo}.")

        state = pr.get("state")
        if state:
            session.note(f"Состояние PR: {state}{', влит' if pr.get('merged') else ''}.")
        session.note(PR_BASE_NOTE)
        if not files and complete:
            limitations.append(EMPTY_FILES_NOTE)

        return self._finalize(
            GitHubContext(
                target_repo=f"{owner}/{name}",
                target_ref=f"PR#{pr_number}",
                base_sha=base_sha,
                head_sha=head_sha,
                files=files,
                limitations=limitations,
                is_partial=not complete,
                base_repo=base_repo,
                head_repo=head_repo,
                revisions_consistent=consistent,
                diagnostics=session.diagnostics,
            )
        )

    def _read_pr_files(
        self, session: "_ReadSession", pr_args: Dict[str, Any], label: str, expected: Optional[int]
    ) -> Tuple[List[FileChange], List[str], bool]:
        collector = _FileCollector(session)
        complete = True
        for page in range(1, MAX_FILE_PAGES + 1):
            try:
                items = session.call_list(
                    "pull_request_read",
                    {"method": "get_files", **pr_args, "page": page, "perPage": PER_PAGE},
                    what=f"{label}: файлы, стр. {page}",
                )
            except GitSourceError as exc:
                collector.notes.append(
                    f"Список файлов PR получен не полностью: страница {page} не прочитана ({exc}). "
                    "Остаток изменений неизвестен."
                )
                complete = False
                break
            collector.add_page(items, page)
            if len(items) < PER_PAGE:
                break
        else:
            collector.notes.append(
                f"Достигнут предел GitHub: прочитано {MAX_FILE_PAGES} страниц по {PER_PAGE} файлов; "
                "остальные файлы PR не получены."
            )
            complete = False

        if expected is not None and len(collector.files) < expected:
            collector.notes.append(
                f"Получено {len(collector.files)} из {expected} изменённых файлов, заявленных GitHub."
            )
            complete = False
        elif expected is not None and len(collector.files) > expected:
            session.note(
                f"Получено файлов больше, чем заявлено GitHub ({len(collector.files)} > {expected}); "
                "PR мог измениться во время чтения."
            )
        return collector.files, collector.notes, complete and not collector.truncated

    # -------------------------------------------------------------- коммит

    def get_commit_context(self, repo: str, commit_sha: str) -> GitHubContext:
        owner, name = parse_repo(repo)
        ref = _validate_ref(commit_sha)

        session = _ReadSession(self)
        repo_args = {"owner": owner, "repo": name}
        label = f"коммит {ref}"

        first = session.call_dict(
            "get_commit",
            {**repo_args, "sha": ref, "detail": "full_patch", "page": 1, "perPage": PER_PAGE},
            what=f"{label}: файлы, стр. 1",
        )
        head_sha = _commit_sha(first, label)
        if not head_sha.startswith(ref.lower()):
            session.note(f"Ссылка «{ref}» разрешена в коммит {head_sha}.")

        files, limitations, complete = self._read_commit_files(session, repo_args, head_sha, label, first)
        parents, parent_notes = self._resolve_parents(session, repo_args, head_sha, label)
        limitations.extend(parent_notes)

        is_merge = parents is not None and len(parents) > 1
        if is_merge:
            session.note(
                f"Merge-коммит, родителей: {len(parents)}; анализ относительно первого родителя {parents[0]}."
            )
            if not self.merge_diff_semantics_verified:
                limitations.append(MERGE_UNVERIFIED_NOTE)
        if not files and complete:
            limitations.append(EMPTY_FILES_NOTE)

        return self._finalize(
            GitHubContext(
                target_repo=f"{owner}/{name}",
                target_ref=ref,
                base_sha=parents[0] if parents else None,
                head_sha=head_sha,
                is_merge_commit=is_merge,
                first_parent_sha=parents[0] if is_merge else None,
                parent_shas=parents,
                files=files,
                limitations=limitations,
                is_partial=not complete,
                diagnostics=session.diagnostics,
            )
        )

    def _read_commit_files(
        self,
        session: "_ReadSession",
        repo_args: Dict[str, Any],
        head_sha: str,
        label: str,
        first_page: Dict[str, Any],
    ) -> Tuple[List[FileChange], List[str], bool]:
        # stats относятся ко всему коммиту; пустые поля GitHub MCP опускает.
        stats = first_page.get("stats")
        expected_lines: Optional[int] = None
        if isinstance(stats, dict):
            expected_lines = _optional_int(stats.get("total")) or 0

        collector = _FileCollector(session)
        complete = True
        page, data = 1, first_page
        while True:
            items = data.get("files") or []
            if not isinstance(items, list):
                collector.notes.append(f"Страница {page} списка файлов коммита имеет неожиданный формат и пропущена.")
                complete = False
                break
            collector.add_page(items, page)
            if len(items) < PER_PAGE:
                break
            if page >= MAX_FILE_PAGES:
                collector.notes.append(
                    f"Достигнут предел GitHub: прочитано {MAX_FILE_PAGES} страниц по {PER_PAGE} файлов; "
                    "остальные файлы коммита не получены."
                )
                complete = False
                break
            page += 1
            try:
                # Дальше запрашиваем по полному SHA: ветка могла сдвинуться.
                data = session.call_dict(
                    "get_commit",
                    {**repo_args, "sha": head_sha, "detail": "full_patch", "page": page, "perPage": PER_PAGE},
                    what=f"{label}: файлы, стр. {page}",
                )
            except GitSourceError as exc:
                collector.notes.append(
                    f"Список файлов коммита получен не полностью: страница {page} не прочитана ({exc}). "
                    "Остаток изменений неизвестен."
                )
                complete = False
                break

        if expected_lines is not None and collector.changed_lines < expected_lines:
            collector.notes.append(
                f"По данным GitHub в коммите изменено строк: {expected_lines}, "
                f"в полученных файлах: {collector.changed_lines}. Часть изменений не получена."
            )
            complete = False
        return collector.files, collector.notes, complete and not collector.truncated

    def _resolve_parents(
        self, session: "_ReadSession", repo_args: Dict[str, Any], head_sha: str, label: str
    ) -> Tuple[Optional[List[str]], List[str]]:
        """Родители через ``get_commit(sha="<sha>^N")``: get_commit их не возвращает.

        None означает «неизвестны». Синтаксис ``^N`` требует живой проверки.
        """
        if not self.resolve_parents:
            return None, [f"Определение родителей отключено. {PARENTS_UNKNOWN_NOTE}"]

        parents: List[str] = []
        for index in range(1, MAX_PARENTS + 1):
            try:
                data = session.call_dict(
                    "get_commit",
                    {**repo_args, "sha": f"{head_sha}^{index}", "detail": "none"},
                    what=f"{label}: родитель №{index}",
                )
                parent = _commit_sha(data, label)
            except GitSourceError as exc:
                absent = exc.kind in (
                    GitSourceErrorKind.INVALID_REVISION,
                    GitSourceErrorKind.NOT_FOUND_OR_NO_ACCESS,
                )
                if index == 1:
                    if absent:
                        return None, [
                            "Родители коммита не определены: GitHub MCP не возвращает родителей, а запрос "
                            f"«<sha>^1» не дал результата (корневой коммит или GitHub не поддерживает такую "
                            f"ссылку). {PARENTS_UNKNOWN_NOTE}"
                        ]
                    return None, [f"Родители коммита не определены ({exc}). {PARENTS_UNKNOWN_NOTE}"]
                if absent:
                    return parents, []
                return parents, [
                    f"Не удалось проверить, есть ли у коммита родитель №{index} ({exc}); "
                    "коммит может оказаться merge-коммитом."
                ]

            if parent == head_sha or parent in parents:
                note = (
                    f"Ответ на запрос «<sha>^{index}» не похож на родителя ({_short(parent)}); "
                    "вероятно, GitHub не поддерживает такую ссылку."
                )
                if index == 1:
                    return None, [f"{note} {PARENTS_UNKNOWN_NOTE}"]
                return parents, [note]
            parents.append(parent)
        return parents, []

    # ------------------------------------------------------------ служебное

    def close(self) -> None:
        """Остановить живой сервер, если клиент его запускал."""
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "GitHubMCPClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _require_transport(self) -> MCPTransport:
        if self._transport is None:
            try:
                self._transport = create_stdio_transport(
                    self.mcp_server_cmd, self._auth_token, call_timeout=self.call_timeout
                )
            except MCPConfigurationError as exc:
                raise GitSourceError(
                    GitSourceErrorKind.NOT_CONFIGURED,
                    f"{exc}; для работы без GitHub используйте MockGitClient или MockGitHubClient",
                ) from None
        return self._transport

    def _finalize(self, ctx: GitHubContext) -> GitHubContext:
        ctx.source = getattr(self._transport, "source", None)
        if ctx.source == SOURCE_MOCK:
            ctx.limitations.insert(0, MOCK_NOTE)
        describe = getattr(self._transport, "describe", None)
        if callable(describe):
            ctx.diagnostics.insert(0, describe())
        ctx.diagnostics.extend(getattr(self._transport, "diagnostics", []))
        ctx.limitations = list(dict.fromkeys(ctx.limitations))
        ctx.diagnostics = list(dict.fromkeys(ctx.diagnostics))
        return ctx


class MockGitHubClient(GitHubMCPClient):
    """Имитация GitHub MCP через настоящий адаптер (без интернета и токенов).

    В отличие от ``MockGitClient`` проходит всю логику адаптера: пагинацию,
    повторы, усечение, определение родителей. По умолчанию любые
    репозиторий, PR и коммит «существуют» и содержат демонстрационные файлы;
    ссылка на коммит, начинающаяся с ``merge_``, имитирует merge-коммит.
    Для сценариев с отказами передайте свой ``FakeGitHub`` (см.
    ``FakeGitHub.fail`` и ``FakeGitHub.on_call``). Каждый результат помечен
    ``source="mock"`` и ограничением ``MOCK_NOTE``.
    """

    def __init__(
        self,
        sample_files: Optional[List[FileChange]] = None,
        *,
        github: Optional[FakeGitHub] = None,
        **kwargs: Any,
    ):
        if github is None:
            files = sample_files if sample_files is not None else _default_sample_files()
            github = FakeGitHub(auto_provision=True, default_files=[_to_fake_file(f) for f in files])
        self.github = github
        kwargs.setdefault("sleep", lambda _seconds: None)
        super().__init__(transport=github.transport(), **kwargs)


class MockGitClient(BaseGitClient):
    """Мок-клиент для локальной автономной разработки и тестов без интернета.

    Простая заглушка с фиксированными ответами, без адаптера и имитации
    GitHub. Ссылка на коммит управляет ответом: префикс ``merge_`` — merge-
    коммит, ``missing_parent``/``no_parent`` — нет первого родителя,
    ``no_base``/``inaccessible_base`` — нет базовой ревизии.
    """

    def __init__(self, sample_files: Optional[List[FileChange]] = None):
        self.sample_files = sample_files or _default_sample_files()

    def get_pr_context(self, repo: str, pr_number: int) -> GitContext:
        return GitContext(
            target_repo=repo,
            target_ref=f"PR#{pr_number}",
            base_sha="base_abc123",
            head_sha="head_def456",
            files=self.sample_files,
            is_partial=False,
        )

    def get_commit_context(self, repo: str, commit_sha: str) -> GitContext:
        is_merge = commit_sha.startswith("merge_")
        is_missing_parent = "missing_parent" in commit_sha or "no_parent" in commit_sha
        is_no_base = "no_base" in commit_sha or "inaccessible_base" in commit_sha

        base_sha = None if is_no_base else ("parent1_111" if is_merge else "parent_000")
        first_parent = None if is_missing_parent else ("parent1_111" if is_merge else None)

        return GitContext(
            target_repo=repo,
            target_ref=commit_sha,
            base_sha=base_sha,
            head_sha=commit_sha,
            is_merge_commit=is_merge,
            first_parent_sha=first_parent,
            files=self.sample_files,
            is_partial=False,
        )


def _default_sample_files() -> List[FileChange]:
    return [
        FileChange(
            filename="src/auth.py",
            status="modified",
            patch="@@ -10,4 +10,6 @@\n+    if password_length < 8:\n+        raise ValueError('Password too short')",
        ),
        FileChange(
            filename="src/config.py",
            status="modified",
            patch="@@ -1,2 +1,3 @@\n+TIMEOUT_SECONDS = 30",
        ),
    ]


# ---------------------------------------------------------------- внутреннее


class _ReadSession:
    """Одно чтение PR или коммита: вызовы с ограниченными повторами и журнал."""

    def __init__(self, client: GitHubMCPClient):
        self._client = client
        self.diagnostics: List[str] = []

    def note(self, text: str) -> None:
        self.diagnostics.append(text)

    def call(self, tool: str, arguments: Dict[str, Any], *, what: str) -> Any:
        _check_allowed(tool, arguments)
        transport = self._client._require_transport()
        delays = self._client.retry_delays
        attempt = 0
        while True:
            attempt += 1
            try:
                result = transport.call_tool(tool, arguments)
            except MCPTimeoutError as exc:
                error = GitSourceError(GitSourceErrorKind.TIMEOUT, str(exc), operation=what)
            except MCPConfigurationError as exc:
                error = GitSourceError(GitSourceErrorKind.NOT_CONFIGURED, str(exc), operation=what)
            except MCPCompatibilityError as exc:
                error = GitSourceError(GitSourceErrorKind.INCOMPATIBLE, str(exc), operation=what)
            except MCPTransportError as exc:
                error = GitSourceError(GitSourceErrorKind.UNAVAILABLE, str(exc), operation=what)
            else:
                if not result.is_error:
                    return _parse_payload(result, what)
                error = classify_tool_error(result.text, operation=what)

            error.attempts = attempt
            if error.retryable and attempt <= len(delays):
                delay = delays[attempt - 1]
                self.note(f"{what}: {error.summary}; повтор через {delay:g} с "
                          f"(попытка {attempt + 1} из {len(delays) + 1}).")
                self._client._sleep(delay)
                continue
            if attempt > 1:
                self.note(f"{what}: не удалось после {attempt} попыток.")
            raise error

    def call_dict(self, tool: str, arguments: Dict[str, Any], *, what: str) -> Dict[str, Any]:
        data = self.call(tool, arguments, what=what)
        if not isinstance(data, dict):
            raise GitSourceError(GitSourceErrorKind.PROTOCOL, "ожидался JSON-объект", operation=what)
        return data

    def call_list(self, tool: str, arguments: Dict[str, Any], *, what: str) -> List[Any]:
        data = self.call(tool, arguments, what=what)
        if not isinstance(data, list):
            raise GitSourceError(GitSourceErrorKind.PROTOCOL, "ожидался JSON-массив", operation=what)
        return data


class _FileCollector:
    """Собирает файлы со страниц: нормализация, дубликаты, усечение, счётчики."""

    def __init__(self, session: _ReadSession):
        self._session = session
        self.files: List[FileChange] = []
        self.notes: List[str] = []
        self.truncated = False
        self.changed_lines = 0
        self._seen: Set[Tuple[str, Optional[str]]] = set()

    def add_page(self, items: List[Any], page: int) -> None:
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("filename"), str) or not item["filename"]:
                self.notes.append(f"На странице {page} пропущена запись о файле без имени.")
                continue
            key = (item["filename"], item.get("previous_filename"))
            if key in self._seen:
                self._session.note(
                    f"Файл {item['filename']} получен повторно (стр. {page}); список мог измениться во время чтения."
                )
                continue
            self._seen.add(key)
            change, notes, truncated, lines = _to_file_change(item)
            self.files.append(change)
            self.notes.extend(notes)
            self.truncated = self.truncated or truncated
            self.changed_lines += lines


def _to_file_change(item: Dict[str, Any]) -> Tuple[FileChange, List[str], bool, int]:
    """Запись файла из ответа MCP → FileChange, примечания, усечён ли, изменённых строк."""
    filename = item["filename"]
    status = str(item.get("status") or "unknown")
    additions = _optional_int(item.get("additions")) or 0
    deletions = _optional_int(item.get("deletions")) or 0
    changes = _optional_int(item.get("changes")) or (additions + deletions)
    patch = item.get("patch") if isinstance(item.get("patch"), str) and item.get("patch") else None
    previous = item.get("previous_filename") or None

    if status in ("renamed", "copied"):
        old_path, new_path = previous, filename
    elif status == "added":
        old_path, new_path = None, filename
    elif status == "removed":
        old_path, new_path = filename, None
    else:
        old_path = new_path = filename

    notes: List[str] = []
    truncated = False
    if patch is None:
        if changes > 0:
            truncated = True
            notes.append(
                f"{filename}: GitHub не передал patch (изменено строк: {changes}; вероятно, слишком большой diff); "
                "изменения файла не проанализированы."
            )
        elif status not in ("renamed", "copied", "unchanged"):
            notes.append(
                f"{filename}: нет patch и счётчиков строк (вероятно, бинарный файл); содержимое не сравнивалось."
            )
    if status in ("renamed", "copied") and previous is None:
        action = "переименовании" if status == "renamed" else "копировании"
        notes.append(f"{filename}: исходный путь при {action} неизвестен (MCP не передал previous_filename).")

    change = FileChange(
        filename=filename,
        status=status,
        old_path=old_path,
        new_path=new_path,
        patch=patch,
        is_truncated=truncated,
    )
    return change, notes, truncated, changes


def _check_allowed(tool: str, arguments: Dict[str, Any]) -> None:
    if tool not in ALLOWED_TOOLS:
        raise PermissionError(f"Адаптеру не разрешён инструмент {tool}: доступ только на чтение.")
    methods = ALLOWED_TOOLS[tool]
    if methods is not None and arguments.get("method") not in methods:
        raise PermissionError(f"Адаптеру не разрешён метод {tool}.{arguments.get('method')}.")


def _parse_payload(result: ToolCallResult, what: str) -> Any:
    text = (result.text or "").strip()
    if not text:
        raise GitSourceError(GitSourceErrorKind.PROTOCOL, "пустой ответ", operation=what)
    try:
        data = json.loads(text)
    except ValueError:
        raise GitSourceError(
            GitSourceErrorKind.PROTOCOL, f"ответ не является JSON: {text[:80]!r}", operation=what
        ) from None
    if data is None:
        raise GitSourceError(GitSourceErrorKind.PROTOCOL, "сервер вернул null", operation=what)
    return data


def _pr_sha(pr: Dict[str, Any], side: str, label: str) -> str:
    branch = pr.get(side)
    sha = branch.get("sha") if isinstance(branch, dict) else None
    if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha.lower()):
        raise GitSourceError(GitSourceErrorKind.PROTOCOL, f"в ответе нет {side}.sha", operation=label)
    return sha.lower()


def _commit_sha(data: Dict[str, Any], label: str) -> str:
    sha = data.get("sha")
    if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha.lower()):
        raise GitSourceError(GitSourceErrorKind.PROTOCOL, "в ответе нет SHA коммита", operation=label)
    return sha.lower()


def _branch_repo(branch: Any) -> Optional[str]:
    repo = branch.get("repo") if isinstance(branch, dict) else None
    name = repo.get("full_name") if isinstance(repo, dict) else None
    return name if isinstance(name, str) and name else None


def _validate_ref(ref: Any) -> str:
    if not isinstance(ref, str) or not _REF_RE.fullmatch(ref.strip()) or ".." in ref:
        raise ValueError(f"Ожидается SHA коммита, имя ветки или тега, получено: {ref!r}")
    return ref.strip()


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _short(sha: str) -> str:
    return sha[:12]


def _to_fake_file(change: FileChange) -> FakeFile:
    return FakeFile(
        filename=change.filename,
        status=change.status,
        patch=change.patch,
        previous_filename=change.old_path if change.status in ("renamed", "copied") else None,
    )
