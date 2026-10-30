"""Tests for the GitHub MCP adapter (Module 3: FR-003, FR-009, FR-013, AC-006, AC-015, TEST-006).

Все проверки идут через имитацию на границе адаптера (FakeGitHub). Они
подтверждают реакцию адаптера на управляемые ответы, но не фактическое
поведение GitHub: его проверяют живые сценарии (AC-014).
"""

import pytest

from specguard.mcp import (
    SOURCE_MOCK,
    FakeFile,
    FakeGitHub,
    GitHubContext,
    GitHubMCPClient,
    GitSourceError,
    GitSourceErrorKind,
    MockGitClient,
    MockGitHubClient,
    classify_tool_error,
    fake_sha,
    parse_repo,
    redact_secrets,
)
from specguard.models.schema import FileChange, GitContext
from specguard.mcp.github_client import (
    EMPTY_FILES_NOTE,
    MERGE_UNVERIFIED_NOTE,
    MOCK_NOTE,
    PER_PAGE,
    _ReadSession,
)

REPO = "acme/shop"


def sha(label: str) -> str:
    return fake_sha("test", label)


def has(lines, fragment: str) -> bool:
    return any(fragment in line for line in lines)


def one_line_files(count: int, prefix: str = "src/mod"):
    return [FakeFile(f"{prefix}{i:04d}.py", patch="@@ -1 +1 @@\n+x = 1") for i in range(count)]


def make_client(github: FakeGitHub, **kwargs):
    sleeps = []
    client = GitHubMCPClient(transport=github.transport(), sleep=sleeps.append, **kwargs)
    return client, sleeps


@pytest.fixture
def github():
    gh = FakeGitHub()
    gh.add_repo(REPO)
    return gh


def add_pr(gh: FakeGitHub, number: int = 7, files=None, **kwargs):
    if files is None:
        files = [
            FakeFile("src/auth.py", patch="@@ -1,2 +1,3 @@\n def login():\n+    check_password()"),
            FakeFile("src/new_name.py", status="renamed", previous_filename="src/old_name.py"),
            FakeFile("src/legacy.py", status="removed", patch="@@ -1,2 +0,0 @@\n-def legacy():\n-    pass"),
            FakeFile("docs/api.md", status="added", patch="@@ -0,0 +1 @@\n+# API"),
        ]
    return gh.add_pull_request(REPO, number, head_sha=sha("head"), base_sha=sha("base"), files=files, **kwargs)


def add_history(gh: FakeGitHub):
    """root ← main-1 ← merge → feature-1 ← root; merge — вершина ветки main."""
    gh.add_commit(REPO, sha("root"), files=[FakeFile("README.md", status="added", patch="@@ -0,0 +1 @@\n+# Shop")])
    gh.add_commit(REPO, sha("main-1"), parents=[sha("root")],
                  files=[FakeFile("src/cart.py", patch="@@ -1 +1,2 @@\n+TAX = 0.2")])
    gh.add_commit(REPO, sha("feature-1"), parents=[sha("root")],
                  files=[FakeFile("src/pay.py", status="added", patch="@@ -0,0 +1 @@\n+def pay(): ...")])
    gh.add_commit(REPO, sha("merge"), parents=[sha("main-1"), sha("feature-1")], branch="main",
                  files=[FakeFile("src/pay.py", status="added", patch="@@ -0,0 +1 @@\n+def pay(): ...")])


# --------------------------------------------------------------------- PR


class TestPullRequest:
    def test_reads_revisions_paths_and_patches(self, github):
        add_pr(github)
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.target_repo == REPO and ctx.target_ref == "PR#7"
        assert ctx.head_sha == sha("head") and ctx.base_sha == sha("base")
        assert ctx.base_repo == REPO and ctx.head_repo == REPO
        assert ctx.revisions_consistent is True
        assert ctx.is_partial is False
        assert ctx.source == SOURCE_MOCK

        files = {f.filename: f for f in ctx.files}
        assert files["src/auth.py"].patch.startswith("@@")
        assert (files["src/new_name.py"].old_path, files["src/new_name.py"].new_path) == (
            "src/old_name.py", "src/new_name.py")
        assert (files["src/legacy.py"].old_path, files["src/legacy.py"].new_path) == ("src/legacy.py", None)
        assert (files["docs/api.md"].old_path, files["docs/api.md"].new_path) == (None, "docs/api.md")
        assert ctx.limitations == [MOCK_NOTE]

    def test_reads_all_pages(self, github):
        add_pr(github, files=one_line_files(250))
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert len(ctx.files) == 250
        assert ctx.is_partial is False
        pages = [args["page"] for args in github.calls_to("pull_request_read", "get_files")]
        assert pages == [1, 2, 3]
        assert all(args["perPage"] == PER_PAGE for args in github.calls_to("pull_request_read", "get_files"))

    def test_exact_page_multiple_stops_on_empty_page(self, github):
        add_pr(github, files=one_line_files(200))
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert len(ctx.files) == 200 and ctx.is_partial is False
        assert len(github.calls_to("pull_request_read", "get_files")) == 3

    def test_fewer_files_than_announced_is_partial(self, github):
        add_pr(github, files=one_line_files(120), changed_files=150)
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.is_partial is True
        assert has(ctx.limitations, "Получено 120 из 150")

    def test_github_listing_cap_is_reported(self):
        gh = FakeGitHub(max_listed_files=3000)
        gh.add_pull_request(REPO, 9, head_sha=sha("h"), base_sha=sha("b"), files=one_line_files(3100))
        client, _ = make_client(gh)

        ctx = client.get_pr_context(REPO, 9)

        assert len(ctx.files) == 3000
        assert ctx.is_partial is True
        assert has(ctx.limitations, "Достигнут предел GitHub")
        assert has(ctx.limitations, "Получено 3000 из 3100")

    def test_missing_patch_for_large_diff_is_truncation(self, github):
        add_pr(github, files=[FakeFile("data/huge.sql", additions=40000, deletions=12)])
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.files[0].patch is None and ctx.files[0].is_truncated is True
        assert ctx.is_partial is True
        assert has(ctx.limitations, "data/huge.sql: GitHub не передал patch")

    def test_binary_file_is_noted_but_not_partial(self, github):
        add_pr(github, files=[FakeFile("img/logo.png")])
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.files[0].is_truncated is False and ctx.is_partial is False
        assert has(ctx.limitations, "img/logo.png: нет patch")

    def test_head_moved_during_read_is_reread(self, github):
        add_pr(github)
        new_head = sha("head-2")

        def push(gh, _tool, _args):
            pr = gh.pull_request(REPO, 7)
            pr.head_sha = new_head
            pr.files = pr.files + [FakeFile("src/late.py", status="added", patch="@@ -0,0 +1 @@\n+late = True")]

        github.on_call("pull_request_read", push, method="get_files", page=1, times=1)
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.head_sha == new_head
        assert ctx.revisions_consistent is True
        assert "src/late.py" in [f.filename for f in ctx.files]
        assert has(ctx.diagnostics, "Head PR изменился")
        assert len(github.calls_to("pull_request_read", "get_files")) == 2

    def test_head_that_keeps_moving_is_inconsistent(self, github):
        add_pr(github)
        counter = iter(range(100))

        def push(gh, _tool, _args):
            gh.pull_request(REPO, 7).head_sha = sha(f"moving-{next(counter)}")

        github.on_call("pull_request_read", push, method="get_files", times=None)
        client, _ = make_client(github, consistency_retries=1)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.revisions_consistent is False
        assert ctx.head_sha == sha("moving-1")
        assert has(ctx.limitations, "Head PR менялся во время чтения")

    def test_failed_recheck_leaves_consistency_unknown(self, github):
        add_pr(github)

        def break_metadata(gh, _tool, _args):
            gh.fail("pull_request_read", method="get", error="server_error", times=None)

        github.on_call("pull_request_read", break_metadata, method="get_files", times=1)
        client, sleeps = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.revisions_consistent is None
        assert len(ctx.files) == 4
        assert has(ctx.limitations, "Не удалось повторно проверить SHA PR")
        assert len(sleeps) == 2

    def test_fork_records_both_repositories(self, github):
        add_pr(github, fork_repo="contrib/shop")
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert (ctx.base_repo, ctx.head_repo) == (REPO, "contrib/shop")
        assert has(ctx.diagnostics, "PR из fork")

    def test_deleted_fork_is_a_limitation(self, github):
        add_pr(github, fork_repo="contrib/shop", fork_deleted=True)
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.head_repo is None
        assert has(ctx.limitations, "fork удалён")

    def test_empty_file_list_is_not_treated_as_no_changes(self, github):
        add_pr(github, files=[])
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.files == []
        assert EMPTY_FILES_NOTE in ctx.limitations

    def test_unknown_pr_is_not_found_or_no_access(self, github):
        client, _ = make_client(github)

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 404)

        assert info.value.kind is GitSourceErrorKind.NOT_FOUND_OR_NO_ACCESS
        assert "нет доступа" in str(info.value)

    def test_inaccessible_repo_looks_the_same_as_missing(self):
        gh = FakeGitHub()
        gh.add_repo(REPO, accessible=False)
        add_pr(gh)  # PR существует, но репозиторий недоступен токену
        client, _ = make_client(gh)

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 7)

        assert info.value.kind is GitSourceErrorKind.NOT_FOUND_OR_NO_ACCESS

    def test_failed_page_gives_partial_result_after_bounded_retries(self, github):
        add_pr(github, files=one_line_files(250))
        github.fail("pull_request_read", method="get_files", page=2, error="server_error", times=None)
        client, sleeps = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert len(ctx.files) == 100
        assert ctx.is_partial is True
        assert has(ctx.limitations, "страница 2 не прочитана")
        assert sleeps == [1.0, 3.0]
        assert has(ctx.diagnostics, "повтор через 1 с")

    def test_rate_limit_is_reported_without_retries(self, github):
        add_pr(github, files=one_line_files(150))
        github.fail("pull_request_read", method="get_files", page=2, error="rate_limit", times=None)
        client, sleeps = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.is_partial is True
        assert has(ctx.limitations, "лимит запросов")
        assert has(ctx.limitations, "Retry after")
        assert sleeps == []

    def test_failed_first_page_is_partial_not_empty(self, github):
        add_pr(github)
        github.fail("pull_request_read", method="get_files", page=1, error="forbidden", times=None)
        client, _ = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.files == [] and ctx.is_partial is True
        assert EMPTY_FILES_NOTE not in ctx.limitations
        assert has(ctx.limitations, "страница 1 не прочитана")

    @pytest.mark.parametrize("number", [0, -1, True, "7"])
    def test_invalid_pr_number(self, github, number):
        client, _ = make_client(github)
        with pytest.raises(ValueError):
            client.get_pr_context(REPO, number)


# ------------------------------------------------------------------ коммит


class TestCommit:
    def test_regular_commit_uses_parent_as_base(self, github):
        add_history(github)
        client, _ = make_client(github)

        ctx = client.get_commit_context(REPO, sha("main-1"))

        assert ctx.head_sha == sha("main-1")
        assert ctx.base_sha == sha("root")
        assert ctx.parent_shas == [sha("root")]
        assert ctx.is_merge_commit is False and ctx.first_parent_sha is None
        assert [f.filename for f in ctx.files] == ["src/cart.py"]
        assert ctx.is_partial is False

    def test_merge_commit_is_compared_with_first_parent(self, github):
        add_history(github)
        client, _ = make_client(github)

        ctx = client.get_commit_context(REPO, sha("merge"))

        assert ctx.is_merge_commit is True
        assert ctx.parent_shas == [sha("main-1"), sha("feature-1")]
        assert ctx.first_parent_sha == sha("main-1") == ctx.base_sha
        assert MERGE_UNVERIFIED_NOTE not in ctx.limitations  # подтверждено живой проверкой
        assert has(ctx.diagnostics, f"первого родителя {sha('main-1')}")

    def test_unverified_mode_brings_the_note_back(self, github):
        add_history(github)
        client, _ = make_client(github, merge_diff_semantics_verified=False)

        ctx = client.get_commit_context(REPO, sha("merge"))

        assert ctx.is_merge_commit is True
        assert MERGE_UNVERIFIED_NOTE in ctx.limitations

    def test_unsupported_parent_syntax_leaves_base_unknown(self):
        gh = FakeGitHub(supports_parent_syntax=False)
        add_history(gh)
        client, _ = make_client(gh)

        ctx = client.get_commit_context(REPO, sha("merge"))

        assert ctx.parent_shas is None and ctx.base_sha is None
        assert ctx.is_merge_commit is False
        assert has(ctx.limitations, "Родители коммита не определены")
        assert has(ctx.limitations, "Базовая ревизия неизвестна")

    def test_ignored_parent_suffix_is_not_taken_for_a_parent(self):
        gh = FakeGitHub(ignores_parent_suffix=True)
        add_history(gh)
        client, _ = make_client(gh)

        ctx = client.get_commit_context(REPO, sha("merge"))

        assert ctx.parent_shas is None and ctx.base_sha is None
        assert ctx.is_merge_commit is False
        assert has(ctx.limitations, "не похож на родителя")

    def test_root_commit_has_unknown_base(self, github):
        add_history(github)
        client, _ = make_client(github)

        ctx = client.get_commit_context(REPO, sha("root"))

        assert ctx.base_sha is None and ctx.parent_shas is None
        assert has(ctx.limitations, "корневой коммит")

    def test_branch_name_is_resolved_and_pinned(self, github):
        github.add_commit(REPO, sha("rel-parent"))
        github.add_commit(REPO, sha("rel"), parents=[sha("rel-parent")], branch="release",
                          files=one_line_files(150))
        client, _ = make_client(github)

        ctx = client.get_commit_context(REPO, "release")

        assert ctx.head_sha == sha("rel") and ctx.target_ref == "release"
        assert len(ctx.files) == 150 and ctx.is_partial is False
        later_refs = [args["sha"] for args in github.calls_to("get_commit")[1:]]
        assert later_refs and all(ref.startswith(sha("rel")) for ref in later_refs)
        assert has(ctx.diagnostics, "Ссылка «release» разрешена")

    def test_stats_reveal_files_missing_from_listing(self):
        gh = FakeGitHub(max_listed_files=100)
        gh.add_commit(REPO, sha("p"))
        gh.add_commit(REPO, sha("big"), parents=[sha("p")], files=one_line_files(150))
        client, _ = make_client(gh)

        ctx = client.get_commit_context(REPO, sha("big"))

        assert len(ctx.files) == 100
        assert ctx.is_partial is True
        assert has(ctx.limitations, "изменено строк: 150")

    def test_unknown_commit_is_invalid_revision(self, github):
        add_history(github)
        client, _ = make_client(github)

        with pytest.raises(GitSourceError) as info:
            client.get_commit_context(REPO, "0" * 40)

        assert info.value.kind is GitSourceErrorKind.INVALID_REVISION

    def test_renamed_file_in_commit_has_unknown_old_path(self, github):
        github.add_commit(REPO, sha("p"))
        github.add_commit(REPO, sha("mv"), parents=[sha("p")],
                          files=[FakeFile("src/b.py", status="renamed", previous_filename="src/a.py")])
        client, _ = make_client(github)

        ctx = client.get_commit_context(REPO, sha("mv"))

        assert ctx.files[0].old_path is None and ctx.files[0].new_path == "src/b.py"
        assert has(ctx.limitations, "исходный путь при переименовании неизвестен")

    def test_failed_second_parent_check_is_reported(self, github):
        add_history(github)
        github.fail("get_commit", sha=f"{sha('merge')}^2", error="server_error", times=None)
        client, sleeps = make_client(github)

        ctx = client.get_commit_context(REPO, sha("merge"))

        assert ctx.base_sha == sha("main-1")
        assert ctx.is_merge_commit is False
        assert has(ctx.limitations, "родитель №2")
        assert len(sleeps) == 2

    def test_empty_merge_commit_is_not_treated_as_no_changes(self, github):
        add_history(github)
        github.add_commit(REPO, sha("empty-merge"), parents=[sha("merge"), sha("feature-1")])
        client, _ = make_client(github)

        ctx = client.get_commit_context(REPO, sha("empty-merge"))

        assert ctx.is_merge_commit is True
        assert EMPTY_FILES_NOTE in ctx.limitations

    @pytest.mark.parametrize("ref", ["", "  ", "HEAD~1", "abc^2", "a b", "main..dev", None])
    def test_invalid_refs_are_rejected(self, github, ref):
        client, _ = make_client(github)
        with pytest.raises(ValueError):
            client.get_commit_context(REPO, ref)


# ------------------------------------------------- транспорт, ошибки, права


class TestTransportAndErrors:
    def test_timeout_is_retried_then_succeeds(self, github):
        add_pr(github)
        github.fail("pull_request_read", method="get", error="timeout", times=1)
        client, sleeps = make_client(github)

        ctx = client.get_pr_context(REPO, 7)

        assert ctx.head_sha == sha("head")
        assert sleeps == [1.0]
        assert has(ctx.diagnostics, "превышено время ожидания")

    def test_timeout_exhausts_bounded_retries(self, github):
        add_pr(github)
        github.fail("pull_request_read", method="get", error="timeout", times=None)
        client, sleeps = make_client(github, retry_delays=(0.5, 0.5, 0.5))

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 7)

        assert info.value.kind is GitSourceErrorKind.TIMEOUT
        assert info.value.attempts == 4
        assert sleeps == [0.5, 0.5, 0.5]
        assert len(github.calls_to("pull_request_read", "get")) == 4

    def test_disconnect_is_unavailable(self, github):
        add_pr(github)
        github.fail("pull_request_read", method="get", error="disconnect", times=None)
        client, _ = make_client(github, retry_delays=())

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 7)

        assert info.value.kind is GitSourceErrorKind.UNAVAILABLE

    def test_non_json_answer_is_protocol_error_without_retries(self, github):
        add_pr(github)
        github.fail("pull_request_read", method="get", error="bad_json", times=None)
        client, sleeps = make_client(github)

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 7)

        assert info.value.kind is GitSourceErrorKind.PROTOCOL
        assert sleeps == []

    def test_bad_credentials(self, github):
        add_pr(github)
        github.fail("pull_request_read", method="get", error="unauthorized")
        client, _ = make_client(github)

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 7)

        assert info.value.kind is GitSourceErrorKind.AUTH_FAILED

    def test_live_client_without_transport_never_falls_back_to_mock(self, monkeypatch):
        for name in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN", "SPECGUARD_MCP_SERVER"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("PATH", "")
        client = GitHubMCPClient()

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 1)

        assert info.value.kind is GitSourceErrorKind.NOT_CONFIGURED
        assert "GITHUB_PERSONAL_ACCESS_TOKEN" in str(info.value)

    @pytest.mark.parametrize("tool, args", [
        ("create_pull_request", {}),
        ("merge_pull_request", {}),
        ("pull_request_read", {"method": "get_review_comments"}),
    ])
    def test_adapter_refuses_tools_outside_read_allowlist(self, github, tool, args):
        client, _ = make_client(github)
        with pytest.raises(PermissionError):
            _ReadSession(client).call(tool, args, what="test")
        assert github.calls == []

    def test_secrets_are_redacted_from_errors(self, github):
        token = "ghp_" + "A1b2C3d4E5" * 4
        add_pr(github)
        github.fail("pull_request_read", method="get", error=f"auth failed for token {token}")
        client, _ = make_client(github)

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context(REPO, 7)

        assert token not in str(info.value)
        assert "[скрыто]" in str(info.value)

    @pytest.mark.parametrize("text, kind", [
        ("failed to get pull request: GET https://api.github.com/repos/a/b/pulls/1: 404 Not Found []",
         GitSourceErrorKind.NOT_FOUND_OR_NO_ACCESS),
        ("failed to get commit: abc: No commit found for SHA: abc", GitSourceErrorKind.INVALID_REVISION),
        ("failed to get pull request: GitHub API rate limit exceeded. Retry after 10m0s.",
         GitSourceErrorKind.RATE_LIMITED),
        ("failed to get pull request files: GitHub secondary rate limit exceeded. Wait before retrying.",
         GitSourceErrorKind.RATE_LIMITED),
        ("failed to get pull request: GET https://api.github.com/repos/a/b/pulls/1: 401 Bad credentials []",
         GitSourceErrorKind.AUTH_FAILED),
        ("failed to get pull request: GET https://api.github.com/repos/a/b/pulls/1: "
         "403 Resource not accessible by personal access token []", GitSourceErrorKind.FORBIDDEN),
        ("failed to get pull request: unexpected status 502: <html>", GitSourceErrorKind.UNAVAILABLE),
        ('failed to get commit: abc: Get "https://api.github.com/repos/a/b/commits/abc": context deadline exceeded',
         GitSourceErrorKind.TIMEOUT),
        ("failed to get commit: abc: GET https://api.github.com:443/repos/a/b/commits/abc: 404 Not Found []",
         GitSourceErrorKind.NOT_FOUND_OR_NO_ACCESS),
        ('failed to get commit: abc: Get "https://api.github.com:443/repos/a/b/commits/abc": '
         "context deadline exceeded", GitSourceErrorKind.TIMEOUT),
        ("missing required parameter: owner", GitSourceErrorKind.TOOL_ERROR),
    ])
    def test_classify_tool_error(self, text, kind):
        assert classify_tool_error(text).kind is kind

    def test_redact_secrets_patterns(self):
        text = redact_secrets("a github_pat_" + "x" * 30 + " b Bearer " + "y" * 30 + " c")
        assert "github_pat_" not in text and "y" * 30 not in text

    @pytest.mark.parametrize("value, expected", [
        ("acme/shop", ("acme", "shop")),
        (" acme/shop.js ", ("acme", "shop.js")),
        ("https://github.com/acme/shop.git", ("acme", "shop")),
        ("github.com/acme/shop/", ("acme", "shop")),
    ])
    def test_parse_repo(self, value, expected):
        assert parse_repo(value) == expected

    @pytest.mark.parametrize("value", ["acme", "acme/shop/extra", "/shop", "acme/..", "", None])
    def test_parse_repo_rejects_garbage(self, value):
        with pytest.raises(ValueError):
            parse_repo(value)


# ------------------------------------------------------------- мок-клиент


class TestMockGitHubClient:
    def test_default_simulation_runs_the_real_adapter(self):
        ctx = MockGitHubClient().get_pr_context("owner/repo", 42)

        assert ctx.target_ref == "PR#42"
        assert [f.filename for f in ctx.files] == ["src/auth.py", "src/config.py"]
        assert ctx.source == SOURCE_MOCK
        assert ctx.limitations[0] == MOCK_NOTE
        assert ctx.revisions_consistent is True

    def test_merge_prefix_simulates_merge_commit(self):
        client = MockGitHubClient()

        merge = client.get_commit_context("owner/repo", "merge_abc")
        regular = client.get_commit_context("owner/repo", "abc123")

        assert merge.is_merge_commit is True
        assert merge.first_parent_sha == merge.base_sha == merge.parent_shas[0]
        assert regular.is_merge_commit is False and regular.base_sha is not None

    def test_custom_sample_files(self):
        client = MockGitHubClient(
            sample_files=[FileChange(filename="app.py", status="added", patch="@@ -0,0 +1 @@\n+x")]
        )
        ctx = client.get_pr_context("owner/repo", 1)

        assert [(f.filename, f.new_path, f.old_path) for f in ctx.files] == [("app.py", "app.py", None)]

    def test_custom_scenario_with_faults(self, github):
        add_pr(github, files=one_line_files(150))
        github.fail("pull_request_read", method="get_files", page=2, error="rate_limit", times=None)

        ctx = MockGitHubClient(github=github).get_pr_context(REPO, 7)

        assert ctx.is_partial is True
        assert ctx.limitations[0] == MOCK_NOTE
        assert has(ctx.limitations, "лимит запросов")

    @pytest.mark.parametrize("ref, method", [
        ("42", "get_pr_context"),
        ("PR#42", "get_pr_context"),
        ("refs/pull/42/head", "get_pr_context"),
        ("merge_abc", "get_commit_context"),
        ("abc123", "get_commit_context"),
    ])
    def test_get_context_dispatches_through_the_adapter(self, ref, method):
        ctx = MockGitHubClient().get_context("owner/repo", ref)

        assert ctx.target_ref.startswith("PR#") is (method == "get_pr_context")
        assert ctx.source == SOURCE_MOCK

    def test_result_is_a_plain_gitcontext_for_the_orchestrator(self):
        ctx = MockGitHubClient().get_commit_context("owner/repo", "merge_abc")

        assert isinstance(ctx, GitHubContext) and isinstance(ctx, GitContext)
        # Общий контракт не расширен: при пересборке в GitContext лишние поля отбрасываются.
        plain = GitContext.model_validate(ctx.model_dump())
        assert plain.first_parent_sha == ctx.first_parent_sha and not hasattr(plain, "parent_shas")


class TestLegacyMockGitClient:
    """Простая заглушка, на которой построены тесты оркестратора: поведение не меняется."""

    def test_fixed_pr_answer(self):
        ctx = MockGitClient().get_pr_context("owner/repo", 42)

        assert (ctx.target_ref, ctx.base_sha, ctx.head_sha) == ("PR#42", "base_abc123", "head_def456")
        assert [f.filename for f in ctx.files] == ["src/auth.py", "src/config.py"]
        assert ctx.limitations == [] and ctx.is_partial is False
        assert type(ctx) is GitContext

    @pytest.mark.parametrize("ref, merge, base, first_parent", [
        ("abc123", False, "parent_000", None),
        ("merge_abc", True, "parent1_111", "parent1_111"),
        ("merge_missing_parent", True, "parent1_111", None),
        ("merge_no_base", True, None, "parent1_111"),
        ("inaccessible_base_1", False, None, None),
    ])
    def test_magic_commit_refs(self, ref, merge, base, first_parent):
        ctx = MockGitClient().get_commit_context("owner/repo", ref)

        assert (ctx.head_sha, ctx.is_merge_commit, ctx.base_sha, ctx.first_parent_sha) == (
            ref, merge, base, first_parent)

    def test_empty_sample_files_fall_back_to_defaults(self):
        assert len(MockGitClient(sample_files=[]).get_pr_context("o/r", 1).files) == 2
