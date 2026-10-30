"""Tests for the live check and the test repository generator (Module 3, part 3: AC-014, TEST-002, TEST-007).

Конвейер проверяется без сети: учебный репозиторий создаётся локально, GitHub
имитируется по нему же. Имитация подтверждает логику проверки, но не
поведение настоящего GitHub — для этого и нужна сама живая проверка.
"""

import json
import shutil
from pathlib import Path

import pytest

from specguard.mcp import FakeFile, MockGitHubClient
from specguard.mcp.live_check import (
    FAIL,
    PASS,
    LocalClone,
    Target,
    TargetResult,
    load_targets,
    main as live_check_main,
    parse_name_status,
    render_markdown,
    run_target,
    simulate_from_clone,
    split_patches,
    summarize,
    write_protocol,
)
from specguard.mcp.make_test_repo import create_test_repo

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="нужен git")

ROOT = Path(__file__).resolve().parents[2]
REPO = "me/specguard-mcp-testrepo"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    for var, value in (("GIT_AUTHOR_NAME", "Test"), ("GIT_AUTHOR_EMAIL", "test@example.com"),
                       ("GIT_COMMITTER_NAME", "Test"), ("GIT_COMMITTER_EMAIL", "test@example.com")):
        patch.setenv(var, value)
    base = tmp_path_factory.mktemp("live")
    info = create_test_repo(base / "repo", REPO, fixture_count=105, large_file_lines=300)
    yield info, load_targets(str(info["targets_path"])), LocalClone(str(base / "repo"))
    patch.undo()


def by_id(targets, scenario_id):
    return next(t for t in targets if t.scenario_id == scenario_id)


def check(result, name):
    return next(c for c in result.checks if c.name == name)


class TestGeneratedRepository:
    def test_history_contains_the_tricky_cases(self, generated):
        info, targets, clone = generated
        merge = info["merge"]

        assert clone.parents(merge) == [info["first_parent"], info["second_parent"]]
        first = {p for _, _, p in clone.changes(info["first_parent"], merge)}
        second = {p for _, _, p in clone.changes(info["second_parent"], merge)}
        assert first == {"src/app.py", "src/auth.py"} and second == {"src/config.py"}
        assert clone.combined_paths(merge) == set()

        regular = {(s, prev, p) for s, prev, p in clone.changes(merge, info["regular"])}
        assert ("renamed", "src/util.py", "src/helpers.py") in regular
        assert ("removed", None, "src/legacy.py") in regular
        assert ("modified", None, "assets/logo.png") in regular

        three_dot = {p for _, _, p in clone.changes("main", "feature/session-timeout", three_dot=True)}
        two_dot = {p for _, _, p in clone.changes("main", "feature/session-timeout")}
        assert len(three_dot) > 100 and "README.md" in two_dot - three_dot
        assert [t.scenario_id for t in targets] == ["MCP-LIVE-PR", "MCP-LIVE-COMMIT", "MCP-LIVE-MANY-FILES",
                                                    "MCP-LIVE-MERGE"]

    def test_refuses_a_non_empty_folder(self, tmp_path):
        (tmp_path / "busy").mkdir()
        (tmp_path / "busy" / "file.txt").write_text("x")

        with pytest.raises(ValueError):
            create_test_repo(tmp_path / "busy", REPO)


class TestLiveCheckOnSimulation:
    def run_all(self, generated, github=None):
        _, targets, clone = generated
        client = MockGitHubClient(github=github or simulate_from_clone(clone, targets))
        return [run_target(client, target, clone) for target in targets]

    def test_everything_is_confirmed_when_github_behaves_as_expected(self, generated, tmp_path):
        results = self.run_all(generated)
        summary = summarize(results)

        assert summary["parent_refs_work"] is True
        assert summary["merge_first_parent"] is True
        assert summary["pr_merge_base"] is True
        assert summary["commit_files_match"] is True
        assert summary["failed_checks"] == 0 and summary["read_errors"] == 0
        assert summary["simulated"] is True
        assert not any("merge_diff_semantics_verified" in item for item in summary["advice"])
        for result in results:  # те же ответы от живого сервера дали бы разрешение включить флаг
            result.context["source"] = "github_mcp"
        assert any("merge_diff_semantics_verified=True" in item for item in summarize(results)["advice"])

        meta = {"started_at": "2026-10-30T10:00:00+03:00", "source": "Имитация", "clone": None,
                "targets_file": None, "specguard_version": "0.1.0", "platform": "test"}
        json_path, md_path = write_protocol(results, summary, meta, tmp_path)
        markdown = md_path.read_text(encoding="utf-8")
        info = generated[0]
        assert "MCP-LIVE-MERGE" in markdown and info["first_parent"][:12] in markdown
        assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["merge_first_parent"] is True

    def test_merge_listed_against_the_second_parent_is_caught(self, generated):
        info, targets, clone = generated
        github = simulate_from_clone(clone, targets)
        github.repos[REPO].commits[info["merge"]].files = clone.fake_files(info["second_parent"], info["merge"])

        merge = run_target(MockGitHubClient(github=github), by_id(targets, "MCP-LIVE-MERGE"), clone)

        assert check(merge, "merge_files").outcome == FAIL
        assert "от второго родителя" in check(merge, "merge_files").detail

    def test_unsupported_parent_refs_are_caught(self, generated):
        _, targets, clone = generated
        github = simulate_from_clone(clone, targets)
        github.supports_parent_syntax = False

        summary = summarize(self.run_all(generated, github))

        assert summary["parent_refs_work"] is False
        assert any("^N" in item for item in summary["advice"])

    def test_wrong_parent_order_is_caught(self, generated):
        info, targets, clone = generated
        github = simulate_from_clone(clone, targets)
        github.repos[REPO].commits[info["merge"]].parents = [info["second_parent"], info["first_parent"]]

        merge = run_target(MockGitHubClient(github=github), by_id(targets, "MCP-LIVE-MERGE"), clone)

        assert check(merge, "parents").outcome == FAIL

    def test_protocol_hides_tokens_from_any_field(self, tmp_path):
        token = "github_pat_" + "Q7" * 20
        result = TargetResult(Target("o/r", "1", title=f"title {token}"), context={
            "head_sha": "a" * 40, "base_sha": None, "first_parent_sha": None, "parent_shas": None,
            "files": 0, "truncated_files": [], "is_partial": False, "limitations": [f"leak {token}"]})

        paths = write_protocol([result], summarize([result]), {
            "started_at": "t", "source": "s", "specguard_version": "0", "platform": "p"}, tmp_path)

        assert all(token not in path.read_text(encoding="utf-8") for path in paths)

    def test_pr_listed_against_the_base_tip_is_caught(self, generated):
        info, targets, clone = generated
        github = simulate_from_clone(clone, targets)
        github.pull_request(REPO, 1).files = clone.fake_files(info["base_tip"], info["pr_tip"])

        result = run_target(MockGitHubClient(github=github), by_id(targets, "MCP-LIVE-PR"), clone)

        assert check(result, "pr_files").outcome == FAIL
        assert "от вершины базовой ветки" in check(result, "pr_files").detail

    def test_wrong_expected_sha_is_an_input_mismatch(self, generated):
        _, targets, clone = generated
        target = by_id(targets, "MCP-LIVE-MERGE")
        wrong = Target(target.repo, target.ref, "X", kind="merge", expected={"expected_first_parent_sha": "0" * 40})

        result = run_target(MockGitHubClient(github=simulate_from_clone(clone, targets)), wrong, clone)

        assert check(result, "revisions").outcome == FAIL
        assert "несовпадение входа" in check(result, "revisions").detail
        assert check(result, "merge_files").outcome == PASS

    def test_read_error_is_recorded_and_secrets_are_hidden(self, generated, tmp_path):
        _, targets, clone = generated
        github = simulate_from_clone(clone, targets)
        token = "ghp_" + "Z9" * 18
        github.fail("pull_request_read", method="get", error=f"bad token {token}", times=None)

        results = [run_target(MockGitHubClient(github=github), t, clone) for t in targets]
        _, md_path = write_protocol(results, summarize(results), {
            "started_at": "t", "source": "s", "specguard_version": "0", "platform": "p"}, tmp_path)

        assert results[0].error_kind == "tool_error" and results[1].error is None
        assert token not in md_path.read_text(encoding="utf-8")


class TestInputsAndParsing:
    @pytest.mark.parametrize("count, text", [(1, "1 файл"), (3, "3 файла"), (11, "11 файлов"),
                                             (105, "105 файлов"), (153, "153 файла"), (121, "121 файл")])
    def test_russian_plural_for_file_counts(self, count, text):
        from specguard.mcp.live_check import _files
        assert _files(count) == text

    def test_reads_participant5_benchmark_format(self):
        targets = load_targets(str(ROOT / "tests" / "ground_truth" / "benchmarks.json"))

        assert [(t.repo, t.ref) for t in targets] == [
            ("sample/demo-app", "PR#101"), ("sample/demo-app", "PR#102"), ("sample/demo-app", "merge_commit_sha123")]
        assert [t.is_pr for t in targets] == [True, True, False]
        assert all(t.expected == {} for t in targets)

    def test_parse_name_status_with_renames(self):
        out = "M\0src/a.py\0R087\0old.py\0new.py\0D\0gone.py\0A\0add.py\0"

        assert parse_name_status(out) == [
            ("modified", None, "src/a.py"), ("renamed", "old.py", "new.py"),
            ("removed", None, "gone.py"), ("added", None, "add.py")]

    def test_split_patches(self):
        diff = "\n".join([
            "diff --git a/new.py b/new.py", "new file mode 100644", "--- /dev/null", "+++ b/new.py",
            "@@ -0,0 +1 @@", "+x = 1",
            "diff --git a/gone.py b/gone.py", "deleted file mode 100644", "--- a/gone.py", "+++ /dev/null",
            "@@ -1 +0,0 @@", "-y = 2",
            "diff --git a/logo.png b/logo.png", "Binary files a/logo.png and b/logo.png differ", "",
        ])

        assert split_patches(diff) == {"new.py": "@@ -0,0 +1 @@\n+x = 1", "gone.py": "@@ -1 +0,0 @@\n-y = 2"}

    def test_markdown_marks_unchecked_verdicts(self):
        text = render_markdown([], summarize([]), {
            "started_at": "t", "source": "s", "specguard_version": "0", "platform": "p"})

        assert "Файлы merge-коммита от первого родителя: не проверялось" in text

    def test_cli_mock_run_writes_protocol(self, tmp_path, capsys):
        code = live_check_main(["--mock", "--repo", "o/r", "--ref", "7", "--ref", "merge_abc", "--out", str(tmp_path)])

        assert code == 0
        assert len(list(tmp_path.glob("mcp-live-*.md"))) == 1
        assert "Протокол:" in capsys.readouterr().out

    def test_cli_requires_targets(self):
        with pytest.raises(SystemExit):
            live_check_main([])

    def test_fake_files_keep_binary_without_patch(self, generated):
        info, _, clone = generated

        files = {f.filename: f for f in clone.fake_files(info["merge"], info["regular"])}

        assert isinstance(files["assets/logo.png"], FakeFile) and files["assets/logo.png"].patch is None
        assert files["src/helpers.py"].previous_filename == "src/util.py" and files["src/helpers.py"].patch
