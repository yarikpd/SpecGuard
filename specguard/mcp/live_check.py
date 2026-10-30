"""Живая проверка адаптера GitHub MCP на настоящем репозитории (Модуль 3, часть 3).

Прогоняет ``GitHubMCPClient`` по целям (PR, обычный коммит, merge-коммит) и
пишет протокол с ревизиями и SHA первого родителя (AC-014, TEST-002,
TEST-007). Если указан локальный клон того же репозитория, ответы GitHub
сверяются с git:
- родители коммита — работает ли запрос ``<sha>^N`` через get_commit;
- список файлов merge-коммита — от какого родителя его строит GitHub;
- список файлов PR — от merge-base или от вершины базовой ветки;
- ожидаемые SHA из файла целей — несовпадение считается ошибкой входа, а не
  адаптера (TEST-007).

Цели читаются из JSON в формате эталонов (``target_repo``, ``target_ref``,
как в tests/ground_truth/benchmarks.json) или из аргументов, поэтому проверка
одинаково работает с учебным репозиторием из ``make_test_repo`` и с
репозиторием участника 5. Ожидания по требованиям ТЗ здесь не читаются.

Запуск:
    python -m specguard.mcp.live_check --targets ЦЕЛИ.json --clone ПУТЬ_К_КЛОНУ
    python -m specguard.mcp.live_check --repo owner/name --ref 1 --ref <sha>
Сервер и токен — из переменных окружения (docs/MCP_LIVE_SETUP.md); токен в
протокол не попадает. ``--mock`` прогоняет то же самое на имитации GitHub.
"""

import argparse
import datetime
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from specguard import __version__
from specguard.mcp.errors import GitSourceError, GitSourceErrorKind, redact_secrets
from specguard.mcp.fake_github import FakeFile, FakeGitHub
from specguard.mcp.github_client import GitHubMCPClient, MockGitHubClient
from specguard.mcp.transport import SOURCE_MOCK

PASS, FAIL, SKIP, INFO = "pass", "fail", "skip", "info"
MARK = {PASS: "✓", FAIL: "✗", SKIP: "—", INFO: "i"}
GIT_STATUS = {"A": "added", "M": "modified", "D": "removed", "R": "renamed", "C": "copied", "T": "changed"}
EXPECTED_KEYS = ("expected_head_sha", "expected_base_sha", "expected_first_parent_sha", "expected_parent_shas")
DEFAULT_OUT = Path.home() / "specguard-live-protocols"
_PR_RE = re.compile(r"^(?:(?:refs/)?pull/|pr\s*#?|#)?(\d+)(?:/head|/merge)?$", re.IGNORECASE)

Change = Tuple[str, Optional[str], str]  # (статус GitHub, прежний путь, путь)


class LiveCheckError(RuntimeError):
    """Проверку нельзя выполнить: нет git, неверный клон, нет целей."""


def pr_number(ref: str) -> Optional[int]:
    """Номер PR по тем же правилам, что и BaseGitClient.get_context."""
    clean = str(ref).strip()
    match = _PR_RE.match(clean)
    prefixed = clean.lower().startswith(("pr", "#", "pull/", "refs/pull/"))
    if match and ((clean.isdigit() and len(clean) <= 7) or prefixed):
        return int(match.group(1))
    return None


@dataclass
class Target:
    repo: str
    ref: str
    scenario_id: str = ""
    title: str = ""
    kind: str = ""
    expected: Dict[str, Any] = field(default_factory=dict)
    base_branch: Optional[str] = None
    head_branch: Optional[str] = None

    @property
    def is_pr(self) -> bool:
        return self.kind == "pr" or (self.kind == "" and pr_number(self.ref) is not None)

    @property
    def label(self) -> str:
        return self.scenario_id or f"{self.repo}@{self.ref}"


@dataclass
class Check:
    name: str
    outcome: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetResult:
    target: Target
    checks: List[Check] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    error_kind: Optional[str] = None
    error: Optional[str] = None


def load_targets(path: str) -> List[Target]:
    """Цели из JSON: список объектов с target_repo и target_ref (или {"targets": [...]})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("targets") if isinstance(data, dict) else data
    targets: List[Target] = []
    seen: Set[Tuple[str, str]] = set()
    for item in items or []:
        if not isinstance(item, dict) or not item.get("target_repo") or not item.get("target_ref"):
            continue
        key = (str(item["target_repo"]).lower(), str(item["target_ref"]).strip())
        if key in seen:
            continue
        seen.add(key)
        targets.append(Target(
            repo=str(item["target_repo"]), ref=str(item["target_ref"]).strip(),
            scenario_id=str(item.get("scenario_id", "")), title=str(item.get("title", "")),
            kind=str(item.get("kind", "")),
            expected={k: item[k] for k in EXPECTED_KEYS if item.get(k)},
            base_branch=item.get("base_branch"), head_branch=item.get("head_branch"),
        ))
    return targets


# ------------------------------------------------------------- локальный git


class LocalClone:
    """Источник истины: локальный клон проверяемого репозитория."""

    def __init__(self, path: str):
        if shutil.which("git") is None:
            raise LiveCheckError("git не найден в PATH")
        self.path = Path(path)
        if self._git("rev-parse", "--is-inside-work-tree", check=False).strip() != "true":
            raise LiveCheckError(f"{path} — не git-репозиторий")

    def _git(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(["git", "-c", "core.quotepath=false", "-C", str(self.path), *args], capture_output=True)
        if check and proc.returncode != 0:
            raise LiveCheckError(f"git {' '.join(args[:2])}: {proc.stderr.decode('utf-8', 'replace').strip()}")
        return proc.stdout.decode("utf-8", errors="replace")

    def repo_name(self) -> Optional[str]:
        url = self._git("remote", "get-url", "origin", check=False).strip()
        match = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
        return f"{match.group(1)}/{match.group(2)}" if match else None

    def resolve(self, ref: str) -> Optional[str]:
        return self._git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False).strip() or None

    def parents(self, sha: str) -> List[str]:
        return self._git("rev-list", "--parents", "-n", "1", sha).split()[1:]

    def changes(self, old: Optional[str], new: str, *, three_dot: bool = False) -> List[Change]:
        if old is None:
            out = self._git("diff-tree", "--root", "--no-commit-id", "-r", "-M", "--name-status", "-z", new)
        else:
            spec = [f"{old}...{new}"] if three_dot else [old, new]
            out = self._git("diff", "-M", "--name-status", "-z", *spec)
        return parse_name_status(out)

    def combined_paths(self, sha: str) -> Set[str]:
        out = self._git("diff-tree", "--cc", "--no-commit-id", "-r", "--name-only", "-z", sha)
        return {path for path in out.split("\0") if path}

    def fake_files(self, old: Optional[str], new: str, *, three_dot: bool = False) -> List[FakeFile]:
        """Файлы в том виде, в каком их отдал бы GitHub: статус, прежний путь, patch."""
        if old is None:
            diff = self._git("show", "--format=", "-M", "--no-color", "--no-ext-diff", new)
        else:
            spec = [f"{old}...{new}"] if three_dot else [old, new]
            diff = self._git("diff", "-M", "--no-color", "--no-ext-diff", *spec)
        patches = split_patches(diff)
        return [
            FakeFile(filename=path, status=status, patch=patches.get(path), previous_filename=previous)
            for status, previous, path in self.changes(old, new, three_dot=three_dot)
        ]


def parse_name_status(out: str) -> List[Change]:
    """Вывод ``git diff --name-status -z`` → [(статус GitHub, прежний путь, путь)]."""
    tokens = out.split("\0")
    changes: List[Change] = []
    i = 0
    while i < len(tokens):
        code = tokens[i]
        if not code:
            i += 1
            continue
        letter = code[0]
        if letter in "RC" and i + 2 < len(tokens):
            changes.append((GIT_STATUS[letter], tokens[i + 1], tokens[i + 2]))
            i += 3
        else:
            changes.append((GIT_STATUS.get(letter, "changed"), None, tokens[i + 1]))
            i += 2
    return changes


def split_patches(diff: str) -> Dict[str, str]:
    """Полный unified diff → {путь: patch начиная с первого @@}, как поле patch у GitHub."""
    patches: Dict[str, str] = {}
    minus = plus = None
    hunk: Optional[List[str]] = None

    def flush() -> None:
        path = plus or minus
        if hunk and path:
            patches[path] = "\n".join(hunk).rstrip("\n")

    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            flush()
            minus = plus = None
            hunk = None
        elif hunk is not None:
            hunk.append(line)
        elif line.startswith("--- a/"):
            minus = line[6:]
        elif line.startswith("+++ b/"):
            plus = line[6:]
        elif line.startswith("@@"):
            hunk = [line]
    flush()
    return patches


def simulate_from_clone(clone: LocalClone, targets: Sequence[Target]) -> FakeGitHub:
    """Имитация GitHub по локальному клону: merge от первого родителя, PR от merge-base."""
    github = FakeGitHub()
    for target in targets:
        if target.repo.lower() not in github.repos:
            github.add_repo(target.repo)
        commits = github.repos[target.repo.lower()].commits
        if target.is_pr:
            base = clone.resolve(target.base_branch or "")
            head = clone.resolve(target.head_branch or "")
            if base and head:
                github.add_pull_request(
                    target.repo, pr_number(target.ref) or 1, head_sha=head, base_sha=base,
                    files=clone.fake_files(base, head, three_dot=True),
                    head_ref=target.head_branch, base_ref=target.base_branch,
                )
            continue
        sha = clone.resolve(target.ref)
        if not sha:
            continue
        parents = clone.parents(sha)
        for parent in parents:
            if parent not in commits:
                github.add_commit(target.repo, parent)
        github.add_commit(target.repo, sha, parents=parents,
                          files=clone.fake_files(parents[0] if parents else None, sha))
    return github


# ------------------------------------------------------------------ проверки


def run_target(client: Any, target: Target, clone: Optional[LocalClone]) -> TargetResult:
    result = TargetResult(target=target)
    try:
        ctx = client.get_context(target.repo, target.ref)
    except GitSourceError as exc:
        result.error_kind, result.error = exc.kind.value, str(exc)
        result.checks.append(Check("read", FAIL, f"чтение не удалось: {exc}"))
        return result

    files = list(ctx.files)
    result.context = {
        "head_sha": ctx.head_sha, "base_sha": ctx.base_sha, "first_parent_sha": ctx.first_parent_sha,
        "parent_shas": getattr(ctx, "parent_shas", None), "is_merge_commit": ctx.is_merge_commit,
        "base_repo": getattr(ctx, "base_repo", None), "head_repo": getattr(ctx, "head_repo", None),
        "revisions_consistent": getattr(ctx, "revisions_consistent", None), "source": getattr(ctx, "source", None),
        "files": len(files), "truncated_files": [f.filename for f in files if f.is_truncated],
        "is_partial": ctx.is_partial, "limitations": list(ctx.limitations),
        "diagnostics": list(getattr(ctx, "diagnostics", [])),
    }
    result.checks.extend(_check_expected(target, ctx))
    if clone is None:
        result.checks.append(Check("git", SKIP, "локальный клон не указан: сверка с git не выполнялась"))
    elif not _clone_applies(clone, target):
        result.checks.append(Check("git", SKIP, f"клон относится к другому репозиторию ({clone.repo_name()})"))
    elif target.is_pr:
        result.checks.append(_check_pr_files(clone, ctx))
    else:
        result.checks.extend(_check_commit(clone, ctx))
    if result.context["truncated_files"]:
        names = ", ".join(result.context["truncated_files"][:5])
        result.checks.append(Check("patch", INFO, f"GitHub не передал patch для файлов: {names}"))
    return result


def _clone_applies(clone: LocalClone, target: Target) -> bool:
    name = clone.repo_name()
    return name is None or name.lower() == target.repo.lower()


def _same(expected: str, actual: Optional[str]) -> bool:
    return bool(actual) and str(actual).lower().startswith(str(expected).lower())


def _check_expected(target: Target, ctx: Any) -> List[Check]:
    if not target.expected:
        return []
    actual = {
        "expected_head_sha": ctx.head_sha,
        "expected_base_sha": ctx.base_sha,
        "expected_first_parent_sha": ctx.first_parent_sha,
    }
    mismatches = []
    for key, value in target.expected.items():
        if key == "expected_parent_shas":
            got = getattr(ctx, "parent_shas", None) or []
            if len(got) != len(value) or not all(_same(e, a) for e, a in zip(value, got, strict=True)):
                mismatches.append(f"родители: ожидались {_shorts(value)}, получены {_shorts(got)}")
        elif not _same(value, actual.get(key)):
            mismatches.append(f"{key[9:]}: ожидался {str(value)[:12]}, получен {str(actual.get(key))[:12]}")
    if mismatches:
        detail = "несовпадение входа (TEST-007), это не ошибка адаптера: " + "; ".join(mismatches)
        return [Check("revisions", FAIL, detail)]
    return [Check("revisions", PASS, "ревизии совпадают с файлом целей")]


def _check_commit(clone: LocalClone, ctx: Any) -> List[Check]:
    head = ctx.head_sha
    if not clone.resolve(head):
        return [Check("git", SKIP, f"коммита {head[:12]} нет в клоне: выполните git fetch")]
    parents = clone.parents(head)
    adapter_parents = getattr(ctx, "parent_shas", None)
    checks: List[Check] = []
    if adapter_parents is None:
        outcome = FAIL if parents else INFO
        checks.append(Check("parents", outcome, f"адаптер не определил родителей; по git: {_shorts(parents) or 'нет'}"))
    elif adapter_parents == parents:
        checks.append(Check("parents", PASS, f"родители совпадают с git: {_shorts(parents)}"))
    else:
        checks.append(Check("parents", FAIL, f"адаптер: {_shorts(adapter_parents)}, git: {_shorts(parents)}"))

    github = {(f.status, f.filename) for f in ctx.files}
    if len(parents) < 2:
        expected = set((s, p) for s, _, p in clone.changes(parents[0] if parents else None, head))
        what = f"git diff {_short(parents[0]) if parents else 'root'}..{_short(head)}"
        checks.append(_compare_files("files", github, expected, what))
        return checks

    variants = {
        "first_parent": _paths(clone.changes(parents[0], head)),
        "second_parent": _paths(clone.changes(parents[1], head)),
        "combined": clone.combined_paths(head),
    }
    checks.append(_detect("merge_files", {p for _, p in github}, variants, "first_parent", {
        "first_parent": "от первого родителя",
        "second_parent": "от второго родителя",
        "combined": "комбинированный diff",
    }))
    return checks


def _check_pr_files(clone: LocalClone, ctx: Any) -> Check:
    base, head = ctx.base_sha, ctx.head_sha
    if not (base and clone.resolve(base) and clone.resolve(head)):
        return Check("pr_files", SKIP, "ревизий PR нет в клоне: выполните git fetch")
    variants = {
        "merge_base": _paths(clone.changes(base, head, three_dot=True)),
        "base_tip": _paths(clone.changes(base, head)),
    }
    return _detect("pr_files", {f.filename for f in ctx.files}, variants, "merge_base", {
        "merge_base": "от общей базы веток (merge-base)", "base_tip": "от вершины базовой ветки",
    })


def _detect(name: str, github: Set[str], variants: Dict[str, Set[str]], wanted: str, words: Dict[str, str]) -> Check:
    matched = [key for key, paths in variants.items() if paths == github]
    data = {"matched": matched, "github_files": len(github), **{f"{k}_files": len(v) for k, v in variants.items()}}
    if matched == [wanted]:
        return Check(name, PASS, f"GitHub строит список файлов {words[wanted]} ({_files(len(github))})", data)
    if wanted in matched:
        return Check(name, INFO, "неоднозначно: варианты дают одинаковый список файлов", data)
    if matched:
        return Check(name, FAIL, "GitHub строит список файлов " + ", ".join(words[k] for k in matched), data)
    return Check(name, FAIL, f"список файлов GitHub ({_files(len(github))}) не совпал ни с одним вариантом git", data)


def _compare_files(name: str, github: Set[Tuple[str, str]], expected: Set[Tuple[str, str]], what: str) -> Check:
    if github == expected:
        return Check(name, PASS, f"файлы и статусы совпадают с {what} ({_files(len(expected))})")
    g_paths, e_paths = {p for _, p in github}, {p for _, p in expected}
    if g_paths == e_paths:
        differ = sorted(f"{p}: GitHub {gs}, git {es}" for gs, p in github for es, q in expected if p == q and gs != es)
        return Check(name, PASS, f"файлы совпадают с {what}, статусы различаются: {'; '.join(differ[:5])}")
    missing, extra = sorted(e_paths - g_paths), sorted(g_paths - e_paths)
    return Check(name, FAIL, f"расхождение с {what}: нет у GitHub {missing[:10]}, лишние {extra[:10]}")


def _files(count: int) -> str:
    tail = count % 100
    if 11 <= tail <= 14:
        word = "файлов"
    else:
        word = {1: "файл", 2: "файла", 3: "файла", 4: "файла"}.get(count % 10, "файлов")
    return f"{count} {word}"


def _paths(changes: Sequence[Change]) -> Set[str]:
    return {path for _, _, path in changes}


def _cell(value: str) -> str:
    return f"`{value}`" if value else "—"


def _short(sha: Optional[str]) -> str:
    return (sha or "")[:12]


def _shorts(shas: Sequence[str]) -> str:
    return ", ".join(_short(s) for s in shas)


# -------------------------------------------------------------------- итоги


def summarize(results: Sequence[TargetResult]) -> Dict[str, Any]:
    def verdict(name: str) -> Optional[bool]:
        checks = [c for r in results for c in r.checks if c.name == name and c.outcome in (PASS, FAIL)]
        return None if not checks else all(c.outcome == PASS for c in checks)

    summary: Dict[str, Any] = {
        "parent_refs_work": verdict("parents"),
        "merge_first_parent": verdict("merge_files"),
        "pr_merge_base": verdict("pr_files"),
        "commit_files_match": verdict("files"),
        "read_errors": sum(1 for r in results if r.error),
        "failed_checks": sum(1 for r in results for c in r.checks if c.outcome == FAIL),
    }
    advice: List[str] = []
    simulated = any(r.context.get("source") == SOURCE_MOCK for r in results)
    summary["simulated"] = simulated
    if simulated:
        advice.append("Это имитация: она проверяет саму проверку, а не поведение GitHub (TEST-006).")
    elif summary["parent_refs_work"] and summary["merge_first_parent"]:
        advice.append("GitHub строит список файлов merge-коммита от первого родителя (AC-014): "
                      "настройка merge_diff_semantics_verified=True обоснована.")
    if summary["parent_refs_work"] is False:
        advice.append("GitHub не понимает «<sha>^N» через get_commit: нужен другой способ получить родителей.")
    if summary["merge_first_parent"] is False:
        advice.append("Список файлов merge-коммита строится не от первого родителя: нужен отдельный способ "
                      "получить diff относительно первого родителя.")
    if summary["pr_merge_base"] is False:
        advice.append("Список файлов PR строится не от merge-base: пересмотреть пометку о базе PR.")
    if any(r.error_kind == GitSourceErrorKind.NOT_CONFIGURED.value for r in results):
        advice.append("Живое подключение не настроено: см. docs/MCP_LIVE_SETUP.md.")
    summary["advice"] = advice
    return summary


def write_protocol(results: Sequence[TargetResult], summary: Dict[str, Any], meta: Dict[str, Any],
                   out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path, md_path = out_dir / f"mcp-live-{stamp}.json", out_dir / f"mcp-live-{stamp}.md"
    payload = {"meta": meta, "summary": summary, "results": [asdict(r) for r in results]}
    json_path.write_text(redact_secrets(json.dumps(payload, ensure_ascii=False, indent=2)) + "\n", encoding="utf-8")
    md_path.write_text(redact_secrets(render_markdown(results, summary, meta)), encoding="utf-8")
    return json_path, md_path


def render_markdown(results: Sequence[TargetResult], summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    def word(value: Optional[bool]) -> str:
        return {True: "подтверждено", False: "НЕ подтверждено", None: "не проверялось"}[value]

    lines = [
        f"# Протокол живой проверки GitHub MCP — {meta['started_at']}",
        "",
        f"- Источник: {meta['source']}",
        f"- Локальный клон: {meta.get('clone') or 'не указан'}",
        f"- Цели: {meta.get('targets_file') or 'из аргументов'}",
        f"- SpecGuard {meta['specguard_version']}, {meta['platform']}",
        "",
        "## Итог",
        "",
        f"- Родители коммита через «<sha>^N»: {word(summary['parent_refs_work'])}",
        f"- Файлы merge-коммита от первого родителя: {word(summary['merge_first_parent'])}",
        f"- Файлы PR от merge-base: {word(summary['pr_merge_base'])}",
        f"- Файлы обычных коммитов совпадают с git: {word(summary['commit_files_match'])}",
        f"- Ошибок чтения: {summary['read_errors']}, проваленных проверок: {summary['failed_checks']}",
    ]
    lines += [f"- {item}" for item in summary["advice"]]
    for result in results:
        target, ctx = result.target, result.context
        lines += ["", f"## {target.label}" + (f" — {target.title}" if target.title else ""), "",
                  f"Цель: `{target.repo}` @ `{target.ref}`"]
        if result.error:
            lines += [f"Ошибка ({result.error_kind}): {result.error}"]
        else:
            lines += [
                f"Ревизии: head {_cell(_short(ctx['head_sha']))}, база {_cell(_short(ctx['base_sha']))}, "
                f"первый родитель {_cell(_short(ctx['first_parent_sha']))}, "
                f"родители {_cell(_shorts(ctx['parent_shas'] or []))}",
                f"Файлов: {ctx['files']}, без patch: {len(ctx['truncated_files'])}, "
                f"частичный: {'да' if ctx['is_partial'] else 'нет'}",
            ]
        lines += ["", "Проверки:"] + [f"- {MARK[c.outcome]} {c.name}: {c.detail}" for c in result.checks]
        if ctx.get("limitations"):
            lines += ["", "Ограничения:"] + [f"- {item}" for item in ctx["limitations"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m specguard.mcp.live_check",
        description="Живая проверка адаптера GitHub MCP с протоколом (AC-014).",
    )
    parser.add_argument("--targets", help="JSON с целями в формате эталонов (target_repo, target_ref)")
    parser.add_argument("--repo", help="репозиторий для целей из --ref: owner/name")
    parser.add_argument("--ref", action="append", default=[], help="PR (номер) или коммит; можно повторять")
    parser.add_argument("--clone", help="локальный клон того же репозитория для сверки с git")
    parser.add_argument("--out", help=f"папка для протокола (по умолчанию {DEFAULT_OUT})")
    parser.add_argument("--mock", action="store_true", help="имитация GitHub вместо живого сервера")
    args = parser.parse_args(argv)
    if args.ref and not args.repo:
        parser.error("--ref требует --repo")

    try:
        targets = load_targets(args.targets) if args.targets else []
        targets += [Target(repo=args.repo, ref=ref) for ref in args.ref]
        if not targets:
            parser.error("укажите --targets или --repo вместе с --ref")
        clone = LocalClone(args.clone) if args.clone else None
    except (LiveCheckError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    if args.mock:
        client: Any = MockGitHubClient(github=simulate_from_clone(clone, targets)) if clone else MockGitHubClient()
    else:
        client = GitHubMCPClient()
    started = datetime.datetime.now().astimezone()
    results: List[TargetResult] = []
    with client:
        for number, target in enumerate(targets, start=1):
            result = run_target(client, target, clone)
            results.append(result)
            failed = [c for c in result.checks if c.outcome == FAIL]
            state = f"ошибка {result.error_kind}" if result.error else ("есть расхождения" if failed else "ok")
            print(f"[{number}/{len(targets)}] {target.label}: {state}")
        describe = getattr(client._transport, "describe", None)
        started_server = callable(describe) and getattr(client._transport, "server_info", None)
        source = "Имитация GitHub (MockGitHubClient)" if args.mock else (
            describe().removeprefix("Источник: ") if started_server else "сервер не запущен")

    summary = summarize(results)
    meta = {
        "started_at": started.isoformat(timespec="seconds"), "source": source,
        "clone": str(Path(args.clone).resolve()) if args.clone else None,
        "targets_file": str(Path(args.targets).resolve()) if args.targets else None,
        "specguard_version": __version__, "platform": platform.platform(),
        "merge_diff_semantics_verified": client.merge_diff_semantics_verified,
    }
    json_path, md_path = write_protocol(results, summary, meta, Path(args.out) if args.out else DEFAULT_OUT)
    print(render_markdown([], summary, meta).split("## Итог", 1)[1].strip())
    print(f"\nПротокол: {md_path}\n          {json_path}")
    if any(r.error_kind == GitSourceErrorKind.NOT_CONFIGURED.value for r in results):
        return 2
    return 1 if summary["failed_checks"] or summary["read_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
