"""GitHub MCP Client and Git Adapters (Module 3)."""

import re
from typing import List, Optional
from abc import ABC, abstractmethod
from specguard.models.schema import GitContext, FileChange


class BaseGitClient(ABC):
    """Абстрактный интерфейс для получения изменений из Git/GitHub."""

    @abstractmethod
    def get_pr_context(self, repo: str, pr_number: int) -> GitContext:
        """Получить контекст PR (файлы, diff, base/head SHA)."""
        pass

    @abstractmethod
    def get_commit_context(self, repo: str, commit_sha: str) -> GitContext:
        """Получить контекст коммита. Для merge-коммита сравнить с первым родителем."""
        pass

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


class MockGitClient(BaseGitClient):
    """Мок-клиент для локальной автономной разработки и тестов без интернета."""

    def __init__(self, sample_files: Optional[List[FileChange]] = None):
        self.sample_files = sample_files or [
            FileChange(
                filename="src/auth.py",
                status="modified",
                patch="@@ -10,4 +10,6 @@\n+    if password_length < 8:\n+        raise ValueError('Password too short')"
            ),
            FileChange(
                filename="src/config.py",
                status="modified",
                patch="@@ -1,2 +1,3 @@\n+TIMEOUT_SECONDS = 30"
            )
        ]

    def get_pr_context(self, repo: str, pr_number: int) -> GitContext:
        return GitContext(
            target_repo=repo,
            target_ref=f"PR#{pr_number}",
            base_sha="base_abc123",
            head_sha="head_def456",
            files=self.sample_files,
            is_partial=False
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
            is_partial=False
        )


class GitHubMCPClient(BaseGitClient):
    """Живой клиент для взаимодействия с локальным GitHub MCP сервером (stdio).
    Участник 3 наполняет этот класс вызовами MCP tools:
    - pull_request_read
    - get_commit
    - get_file_contents
    """

    def __init__(self, mcp_server_cmd: Optional[str] = None, auth_token: Optional[str] = None):
        self.mcp_server_cmd = mcp_server_cmd
        self.auth_token = auth_token

    def get_pr_context(self, repo: str, pr_number: int) -> GitContext:
        # TODO (Участник 3): Вызов инструмента pull_request_read через MCP
        # В случае недоступности MCP можно сфоллбэчиться на Mock
        return MockGitClient().get_pr_context(repo, pr_number)

    def get_commit_context(self, repo: str, commit_sha: str) -> GitContext:
        # TODO (Участник 3): Вызов get_commit, получение 1-го родителя для merge-коммита
        return MockGitClient().get_commit_context(repo, commit_sha)
