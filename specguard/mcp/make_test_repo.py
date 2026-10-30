"""Учебный репозиторий для живой проверки адаптера GitHub MCP (Модуль 3, часть 3).

Создаёт локальный git-репозиторий с историей, на которой видны спорные места
GitHub MCP:
- merge-коммит, у которого изменения относительно первого и второго родителя
  различаются (проверка FR-003, AC-014);
- обычный коммит с переименованием, удалением и изменением бинарного файла;
- ветка для PR, база которой ушла вперёд после ответвления (merge-base и
  вершина базы дают разные списки файлов), с числом файлов больше 100
  (постраничная выдача) и большим файлом (GitHub может не передать patch).

Рядом с репозиторием пишет файл целей для ``live_check`` в формате эталонов
участника 5 (``target_repo``, ``target_ref``) с ожидаемыми SHA (TEST-007).
Ожидаемых выводов анализа в репозитории нет (TEST-004).

Запуск:
    python -m specguard.mcp.make_test_repo ПАПКА --repo ЛОГИН/specguard-mcp-testrepo
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PR_BRANCH = "feature/session-timeout"
MERGED_BRANCH = "feature/password-policy"
FIXTURE_COUNT = 150
LARGE_FILE_LINES = 20000

SPEC = """# ТЗ учебного приложения

- FR-001: Пароль пользователя должен содержать не менее 8 символов.
- FR-002: Пароль должен содержать хотя бы одну цифру.
- NFR-001: Таймаут сетевого запроса не превышает 30 секунд.
- NFR-002: Сессия пользователя истекает после 15 минут бездействия.
"""


UTIL = "".join(
    f'def scale_{n}(value):\n    """Умножить значение на {n}."""\n    return value * {n}\n\n\n'
    for n in range(2, 8)
) + "def clamp(value, low, high):\n    return max(low, min(value, high))\n"


class RepoError(RuntimeError):
    """Не удалось выполнить команду git."""


def _png(variant: int) -> bytes:
    # Заголовок PNG и нулевые байты: git и GitHub считают файл бинарным.
    return b"\x89PNG\r\n\x1a\n" + bytes([0, 0, 0, 13, variant]) + b"\x00" * 32


def create_test_repo(
    path: Union[str, Path],
    repo: str,
    *,
    targets_path: Optional[Union[str, Path]] = None,
    fixture_count: int = FIXTURE_COUNT,
    large_file_lines: int = LARGE_FILE_LINES,
) -> Dict[str, Any]:
    """Создать репозиторий в пустой папке ``path`` и файл целей; вернуть SHA и пути."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"папка {path} не пуста")
    path.mkdir(parents=True, exist_ok=True)
    targets_file = Path(targets_path) if targets_path else path.parent / f"{path.name}-targets.json"

    git = _Git(path)
    git("init", "-q", "-b", "main")

    root = git.commit("Initial app skeleton", {
        "README.md": "# Учебное приложение SpecGuard\n",
        "docs/spec.md": SPEC,
        "src/app.py": "def main():\n    print('app')\n",
        "src/util.py": UTIL,
        "src/legacy.py": "def old_login(user):\n    return True\n",
        "assets/logo.png": _png(1),
    })

    git("switch", "-q", "-c", MERGED_BRANCH)
    git.commit("Add password policy", {
        "src/auth.py": "MIN_PASSWORD_LENGTH = 8\n\n\ndef validate(password):\n"
                       "    return len(password) >= MIN_PASSWORD_LENGTH\n",
        "src/app.py": "from auth import validate\n\n\ndef main():\n    print(validate('secret123'))\n",
    })
    feature_tip = git.commit("Require a digit in passwords", {
        "src/auth.py": "MIN_PASSWORD_LENGTH = 8\n\n\ndef validate(password):\n"
                       "    return len(password) >= MIN_PASSWORD_LENGTH and any(c.isdigit() for c in password)\n",
    })

    git("switch", "-q", "main")
    main_before_merge = git.commit("Add request timeout setting", {
        "src/config.py": "TIMEOUT_SECONDS = 30\n",
    })
    git("merge", "--no-ff", "-q", "-m", f"Merge branch '{MERGED_BRANCH}'", MERGED_BRANCH)
    merge = git.head()

    git("mv", "src/util.py", "src/helpers.py")
    git("rm", "-q", "src/legacy.py")
    regular = git.commit("Rename util module, drop legacy code, refresh logo", {
        "src/helpers.py": UTIL + "\n\ndef is_blank(text):\n    return not text.strip()\n",
        "assets/logo.png": _png(2),
    })

    git("switch", "-q", "-c", PR_BRANCH)
    pr_first = git.commit("Add session timeout", {
        "src/config.py": "TIMEOUT_SECONDS = 30\nSESSION_IDLE_MINUTES = 15\n",
        "src/session.py": "from config import SESSION_IDLE_MINUTES\n\n\ndef expired(idle_minutes):\n"
                          "    return idle_minutes >= SESSION_IDLE_MINUTES\n",
    })
    fixtures: Dict[str, Union[str, bytes, None]] = {
        f"fixtures/case_{i:03d}.json": json.dumps({"case": i, "password": f"secret{i}"}) + "\n"
        for i in range(fixture_count)
    }
    if large_file_lines:
        fixtures["data/large_fixture.csv"] = "id,value\n" + "".join(
            f"{i},{i * 7919 % 100003}\n" for i in range(large_file_lines)
        )
    pr_tip = git.commit("Add generated test fixtures", fixtures)

    git("switch", "-q", "main")
    base_tip = git.commit("Describe configuration in README", {
        "README.md": "# Учебное приложение SpecGuard\n\nНастройки лежат в src/config.py.\n",
    })

    targets: List[Dict[str, Any]] = [
        {
            "scenario_id": "MCP-LIVE-PR",
            "title": "PR из ветки, база которой ушла вперёд; больше 100 файлов и большой файл",
            "target_repo": repo, "target_ref": "PR#1", "kind": "pr",
            "base_branch": "main", "head_branch": PR_BRANCH,
            "expected_head_sha": pr_tip, "expected_base_sha": base_tip,
        },
        {
            "scenario_id": "MCP-LIVE-COMMIT",
            "title": "Обычный коммит: переименование, удаление, бинарный файл",
            "target_repo": repo, "target_ref": regular, "kind": "commit",
            "expected_parent_shas": [merge],
        },
        {
            "scenario_id": "MCP-LIVE-MANY-FILES",
            "title": "Коммит больше чем на 100 файлов: постраничная выдача",
            "target_repo": repo, "target_ref": pr_tip, "kind": "commit",
            "expected_parent_shas": [pr_first],
        },
        {
            "scenario_id": "MCP-LIVE-MERGE",
            "title": "Merge-коммит: изменения от первого и второго родителя различаются",
            "target_repo": repo, "target_ref": merge, "kind": "merge",
            "expected_first_parent_sha": main_before_merge,
            "expected_parent_shas": [main_before_merge, feature_tip],
        },
    ]
    targets_file.write_text(json.dumps(targets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "path": path, "targets_path": targets_file, "root": root, "merge": merge, "regular": regular,
        "pr_first": pr_first, "pr_tip": pr_tip, "base_tip": base_tip,
        "first_parent": main_before_merge, "second_parent": feature_tip,
    }


class _Git:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, *args: str) -> str:
        proc = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True)
        if proc.returncode != 0:
            message = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RepoError(f"git {' '.join(args[:3])}: {message}")
        return proc.stdout.decode("utf-8", errors="replace")

    def head(self) -> str:
        return self("rev-parse", "HEAD").strip()

    def commit(self, message: str, files: Dict[str, Union[str, bytes, None]]) -> str:
        for rel, content in files.items():
            target = self.path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content if isinstance(content, bytes) else str(content).encode("utf-8"))
        self("add", "-A")
        self("commit", "-q", "-m", message)
        return self.head()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m specguard.mcp.make_test_repo",
        description="Создать учебный репозиторий для живой проверки адаптера GitHub MCP.",
    )
    parser.add_argument("path", help="новая пустая папка для репозитория")
    parser.add_argument("--repo", required=True, help="будущий репозиторий на GitHub: ЛОГИН/ИМЯ")
    parser.add_argument("--targets", help="куда записать файл целей (по умолчанию рядом с папкой)")
    parser.add_argument("--no-large-file", action="store_true", help="не добавлять большой файл")
    args = parser.parse_args(argv)
    try:
        info = create_test_repo(args.path, args.repo, targets_path=args.targets,
                                large_file_lines=0 if args.no_large_file else LARGE_FILE_LINES)
    except (ValueError, RepoError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    path, targets, repo = info["path"].resolve(), info["targets_path"].resolve(), args.repo
    print(f"Репозиторий создан: {path}")
    print(f"Файл целей: {targets}")
    print()
    print("Дальше:")
    print(f"1. Создайте на GitHub пустой репозиторий {repo} (без README и .gitignore).")
    print(f'2. git -C "{path}" remote add origin https://github.com/{repo}.git')
    print(f'   git -C "{path}" push -u origin main {PR_BRANCH}')
    print(f"3. Откройте https://github.com/{repo}/compare/main...{PR_BRANCH} и создайте PR (он получит номер #1).")
    print(f'4. python -m specguard.mcp.live_check --targets "{targets}" --clone "{path}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
