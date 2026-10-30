"""SpecGuard's interactive terminal and scriptable commands."""
import os
from pathlib import Path

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from specguard.cli.catalog import GitHubCatalog
from specguard.mcp.github_client import parse_repo
from specguard.cli.config import APIConfig, discover_models, load_config, load_github_token, load_mcp_server, save_mcp_server, save_api_key, save_config
from specguard.cli.history import HistoryStore
from specguard.cli.picker import SelectionCancelled, choose
from specguard.cli.render import render_report
from specguard.cli.viewer import ReportViewer
from specguard.cli.repositories import saved_repositories, save_repository, delete_repository
from specguard.cli.service import check_model_connection, review, run_audit

app = typer.Typer(help="Проверка изменений GitHub по ТЗ.", no_args_is_help=False)
console = Console(style="#eadde0")
history_store = HistoryStore()


def _error(exc):
    console.print(Text(f"Ошибка: {exc}", style="bold #f17d8b"))


def _choose(title, options, extra=None, searchable=False, **kwargs):
    return choose(title, options, extra, searchable=searchable, **kwargs)


def _catalog_pick(kind, repo=None):
    catalog = GitHubCatalog()
    try:
        page = 1
        query = ""
        while True:
            console.print("Загрузка репозиториев…" if kind == "repo" else "Загрузка pull requests…")
            items = catalog.repositories(page, query) if kind == "repo" else catalog.pull_requests(repo, page)
            saved = saved_repositories() if kind == "repo" else []
            options = [(f"★ {name}", name) for name in saved]
            options.extend((label, value) for label, value in items if value.casefold() not in {name.casefold() for name in saved})
            if not items:
                console.print("Репозитории не найдены по выбранному запросу." if kind == "repo" else "Открытых PR нет.")
            if kind == "repo":
                options.append(("Вставить ссылку на публичный репозиторий", "__url__"))
                options.append(("Поиск репозиториев…", "__search__"))
                if query:
                    options.append(("Мои репозитории", "__mine__"))
            if page > 1:
                options.append(("← Предыдущая страница", "__previous__"))
            if len(items) == 50:
                options.append(("Следующая страница →", "__next__"))
            result = _choose("Репозитории" if kind == "repo" else "Открытые pull requests", options, "Ввести вручную",
                             deletable=saved, on_delete=delete_repository if kind == "repo" else None)
            if result == "__url__":
                while True:
                    url = Prompt.ask("Ссылка GitHub, например https://github.com/owner/repo (пусто — назад)", default="")
                    if not url.strip():
                        break
                    try:
                        owner, name = parse_repo(url)
                        return save_repository(f"{owner}/{name}")
                    except ValueError:
                        _error("Укажите ссылку на репозиторий GitHub: https://github.com/owner/repo")
            elif result == "__search__":
                query = Prompt.ask("Поиск GitHub (пусто — мои репозитории)", default="")
                page = 1
            elif result == "__mine__":
                query = ""
                page = 1
            elif result == "__next__":
                page += 1
            elif result == "__previous__":
                page -= 1
            else:
                return result
    finally:
        catalog.close()


def _select_specs(start=None):
    directory = Path(start or Path.cwd()).resolve()
    selected = []
    while True:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError as exc:
            _error(exc)
            directory = directory.parent
            continue
        options = [("📁 ..", directory.parent)]
        options += [(f"📁 {item.name}", item) for item in entries if item.is_dir()]
        options += [(f"📄 {item.name}", item) for item in entries
                    if item.is_file() and item.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}]
        chosen = _choose(f"Выбор ТЗ · {directory}", options, "Отмена", searchable=True)
        if chosen is None:
            raise SelectionCancelled()
        if chosen.is_dir():
            directory = chosen
            continue
        if chosen not in selected:
            selected.append(chosen)
        console.print(Panel(Text("\n".join(str(p) for p in selected)), title="Выбранные файлы ТЗ"))
        action = _choose("Файлы ТЗ", [("Подтвердить и продолжить", "done"), ("Добавить ещё файл", "add")], "Отмена")
        if action is None:
            raise SelectionCancelled()
        if action == "done":
            return [str(path) for path in selected]


def _choose_audit_specs():
    source = _choose("Откуда взять ТЗ?", [("Выбрать файлы", "files"), ("Из предыдущей проверки", "history")], "Назад")
    if source is None:
        raise SelectionCancelled()
    if source == "files":
        return _select_specs()
    reports = [report for report in history_store.list() if report.metadata.get("spec_snapshots")]
    audit_id = _choose("Выберите предыдущую проверку", [
        (f"{report.timestamp} · {report.target_repo}@{report.target_ref} · {report.audit_id}", report.audit_id)
        for report in reports], "Назад")
    if audit_id is None:
        raise SelectionCancelled()
    report = next(report for report in reports if report.audit_id == audit_id)
    specs = list(report.metadata["spec_snapshots"])
    console.print(Panel(Text("\n".join(item["path"] for item in specs)), title="Сохранённые ТЗ"))
    action = _choose("Использовать сохранённые ТЗ?", [("Подтвердить", "done"), ("Добавить новые файлы", "add")], "Назад")
    if action is None:
        raise SelectionCancelled()
    if action == "add":
        specs.extend(_select_specs())
    return specs


def _select_mcp_server(start=None):
    current = load_mcp_server()
    directory = Path(start or (Path(current).parent if current else Path.home())).expanduser().resolve()
    while True:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
            options = [("📁 ..", directory.parent)]
            options += [(f"📁 {item.name}", item) for item in entries if item.is_dir()]
            options += [(f"⚙ {item.name}", item) for item in entries if item.is_file()]
        except OSError as exc:
            _error(exc)
            directory = directory.parent
            continue
        options.append(("Искать github-mcp-server в PATH", "__path__"))
        chosen = _choose(f"Исполняемый файл MCP · {directory}", options, "Отмена", searchable=True)
        if chosen is None:
            raise SelectionCancelled()
        if chosen == "__path__":
            return ""
        if chosen.is_dir():
            directory = chosen
        elif not chosen.is_file() or not os.access(chosen, os.X_OK):
            _error("Файл не является исполняемым. Выберите программу с разрешением на запуск.")
        else:
            return str(chosen.resolve())


def _github_settings():
    while True:
        choice = _choose("GitHub MCP", [
            ("Токен GitHub · " + ("задан" if load_github_token() else "не задан"), "token"),
            ("Путь к серверу · " + (load_mcp_server() or "поиск в PATH"), "server"),
            ("Проверить подключение", "test"),
        ], "Назад")
        if choice is None:
            return
        if choice == "token":
            token = typer.prompt("GitHub personal access token (пустой ввод очищает)", hide_input=True, default="", show_default=False)
            save_api_key("github", token)
            os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = token
            if not token:
                os.environ.pop("GITHUB_TOKEN", None)
            console.print("Токен сохранён локально. Права: Contents и Pull requests — только чтение.")
        elif choice == "server":
            try:
                path = _select_mcp_server()
            except SelectionCancelled:
                continue
            save_mcp_server(path)
            if path:
                os.environ["SPECGUARD_MCP_SERVER"] = path
            else:
                os.environ.pop("SPECGUARD_MCP_SERVER", None)
        else:
            try:
                catalog = GitHubCatalog()
                try:
                    account = catalog._call("get_me", {})
                    console.print(Text(f"Подключено: {account.get('login', 'GitHub')}", style="#80d7a1"))
                finally:
                    catalog.close()
            except Exception as exc:
                _error(exc)


def _configure():
    config = load_config()
    console.print(Panel("Адрес, модель и ключ сохраняются локально. Переменные окружения имеют приоритет над сохранёнными настройками.", title="Настройки моделей", border_style="#c6afb6"))
    while True:
        kind = _choose("Какую модель изменить?", [
            (f"Для проверки · {config.llm.model or 'не настроена'}", "llm"),
            (f"Для embedding · {config.embedding.model or 'не настроена'}", "embedding"),
        ], "Назад")
        if kind is None:
            return
        current = getattr(config, kind)
        name = "Модель для проверки" if kind == "llm" else "Модель для embedding"
        while True:
            try:
                field = _choose(name, [
                    (f"API URL · {current.base_url or 'не указан'}", "url"),
                    (f"Модель · {current.model or 'не указана'}", "model"),
                    (f"API-ключ · {'задан' if current.api_key else 'не задан'}", "key"),
                    ("Проверить подключение", "test"),
                ], "Назад")
                if field is None:
                    break
                if field == "test":
                    console.print("Проверка подключения…")
                    try:
                        result = check_model_connection(kind, current)
                        console.print(Text(result, style="#80d7a1"))
                    except Exception as exc:
                        message = str(exc)
                        if current.api_key:
                            message = message.replace(current.api_key, "[скрыто]")
                        _error(message)
                elif field == "key":
                    key = typer.prompt("API-ключ (пустой ввод очищает ключ)", hide_input=True, default="", show_default=False)
                    save_api_key(kind, key)
                    current.api_key = key
                    os.environ[f"SPECGUARD_{kind.upper()}_API_KEY"] = key
                    console.print("[#80d7a1]Ключ сохранён локально.[/#80d7a1]")
                elif field == "url":
                    previous = current.base_url
                    suffix = "/chat/completions" if kind == "llm" else "/embeddings"
                    console.print(Text("Для Gemini: https://generativelanguage.googleapis.com/v1beta/openai/" + ("chat/completions" if kind == "llm" else "embeddings"), style="#c6afb6"))
                    console.print(Text(f"Укажите полный URL конечной точки. Для OpenAI-совместимого API обычно заканчивается на {suffix}.", style="#c6afb6"))
                    current.base_url = Prompt.ask("API URL", default=current.base_url or f"http://localhost:11434/v1{suffix}")
                    try:
                        save_config(config, kind)
                    except Exception:
                        current.base_url = previous
                        raise
                    console.print("[#80d7a1]Адрес сохранён.[/#80d7a1]")
                else:
                    try:
                        models = discover_models(current)
                        selected = _choose(f"{name}: доступные модели", [(m, m) for m in models], "Ввести вручную")
                    except SelectionCancelled:
                        continue
                    except Exception as exc:
                        console.print(Text(f"Список моделей недоступен: {exc}", style="#e2bd89"))
                        selected = None
                    current.model = selected or Prompt.ask("Модель", default=current.model or None)
                    save_config(config, kind)
                    console.print("[#80d7a1]Модель сохранена.[/#80d7a1]")
            except SelectionCancelled:
                break
            except ValueError as exc:
                _error(exc)


def _show_history():
    reports = history_store.list()
    if not reports:
        console.print("История пуста.")
        return
    table = Table(title="История проверок")
    for title in ("ID", "Дата", "Репозиторий", "PR / SHA", "Замечания"):
        table.add_column(title)
    for report in reports:
        table.add_row(report.audit_id, report.timestamp, report.target_repo, report.target_ref, str(len(report.findings)))
    console.print(table)


def _display_report(content):
    if console.is_terminal:
        ReportViewer(content).run()
    else:
        console.print(content)


def _show(audit_id):
    report = history_store.get(audit_id)
    if report is None:
        raise ValueError("Аудит не найден")
    parts = [render_report(report)]
    for entry in history_store.list_disputes(audit_id):
        r, s = entry["request"], entry["response"]
        parts.append(Panel(Text(f"{s['timestamp']} · {r['finding_req_id']}\n{r['user_comment']}\n{s['dispute_status']}\n{s['reasoning']}"), title="Пересмотр"))
    _display_report(Group(*parts))


def _dispute(audit_id, req_id, comment, demo):
    response, path = review(load_config(), history_store, audit_id, req_id, comment, demo)
    console.print(Panel(Text(f"{response.dispute_status}\n{response.reasoning}\nНовый статус: {response.updated_finding.status.value}"), title="Результат пересмотра"))
    console.print(Text(f"Сохранено: {path}", style="#80d7a1"))


def _interactive():
    console.clear()
    console.print(Panel("Аудит изменений по вашему ТЗ", title="[bold #eadde0]SpecGuard[/bold #eadde0]", subtitle="GitHub · RAG · Review", border_style="#c6afb6"))
    while True:
        try:
            choice = _choose("Что сделать?", [("Новая проверка", "audit"), ("История и пересмотр", "history"), ("Настройки моделей", "config"), ("Настройки GitHub MCP", "github")], "Выход")
        except SelectionCancelled:
            return
        if choice is None:
            return
        try:
            if choice == "config":
                _configure()
            elif choice == "github":
                _github_settings()
            elif choice == "audit":
                spec = _choose_audit_specs()
                mode = _choose("Источник данных", [("GitHub MCP", "live"), ("Демонстрация на моках", "demo")])
                demo = mode == "demo"
                repo = None
                if not demo:
                    try:
                        repo = _catalog_pick("repo")
                    except SelectionCancelled:
                        raise
                    except Exception as exc:
                        _error(exc)
                repo = repo or Prompt.ask("Репозиторий owner/repo")
                target = None
                if not demo and _choose("Объект проверки", [("Выбрать открытый PR", "pr"), ("Ввести PR или SHA", "manual")]) == "pr":
                    try:
                        target = _catalog_pick("pr", repo)
                    except SelectionCancelled:
                        raise
                    except Exception as exc:
                        _error(exc)
                target = target or Prompt.ask("Номер PR или SHA коммита")
                summary = Prompt.ask("Что выполнено", default="")
                lang = _choose("Язык отчёта", [("Русский", "ru"), ("English", "en")])
                report = run_audit(load_config(), history_store, spec, repo, target, summary, lang, demo, lambda msg: console.print(Text(msg, style="#c6afb6")))
                _display_report(render_report(report))
            else:
                reports = history_store.list()
                selected = _choose("История", [(f"{r.timestamp} · {r.target_repo}@{r.target_ref} · {r.audit_id}", r.audit_id) for r in reports], "Назад")
                if selected:
                    _show(selected)
                    if _choose("Действие", [("Оспорить замечание", "dispute")], "Назад"):
                        report = history_store.get(selected)
                        req = _choose("Замечание", [(f"{f.req_id} · {f.title}", f.req_id) for f in report.findings], "Назад")
                        if req:
                            _dispute(selected, req, Prompt.ask("Комментарий"), report.metadata.get("mode") == "demo")
        except SelectionCancelled:
            continue
        except (Exception, KeyboardInterrupt) as exc:
            _error(exc)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Запустите без аргументов для интерактивного меню."""
    if ctx.invoked_subcommand is None:
        _interactive()


@app.command()
def audit(spec: str = typer.Option(..., "--spec", "-s"), repo: str = typer.Option(..., "--repo", "-r"),
          target: str = typer.Option(..., "--target", "-t"), summary: str = typer.Option("", "--summary", "-m"),
          lang: str = typer.Option("ru", "--lang", "-l"), demo: bool = typer.Option(False, "--demo"),
          llm_url: str = typer.Option(None, "--llm-url"), llm_model: str = typer.Option(None, "--llm-model"),
          llm_api_key: str = typer.Option(None, "--llm-api-key"), embedding_url: str = typer.Option(None, "--embedding-url"),
          embedding_model: str = typer.Option(None, "--embedding-model"), embedding_api_key: str = typer.Option(None, "--embedding-api-key")):
    """Запустить проверку PR или коммита."""
    try:
        config = load_config()
        for obj, url, model, key in ((config.llm, llm_url, llm_model, llm_api_key), (config.embedding, embedding_url, embedding_model, embedding_api_key)):
            if url is not None: obj.base_url = url
            if model is not None: obj.model = model
            if key is not None: obj.api_key = key
        report = run_audit(config, history_store, spec, repo, target, summary, lang, demo, lambda msg: console.print(Text(msg, style="#c6afb6")))
        _display_report(render_report(report))
    except Exception as exc:
        _error(exc)
        raise typer.Exit(1)


@app.command("history")
def history():
    """Показать историю проверок."""
    _show_history()


@app.command()
def show(audit_id: str):
    """Показать сохранённый отчёт и пересмотры."""
    try:
        _show(audit_id)
    except Exception as exc:
        _error(exc)
        raise typer.Exit(1)


@app.command()
def dispute(audit_id: str, req_id: str = typer.Option(..., "--req", "-r"), comment: str = typer.Option(..., "--comment", "-c"), demo: bool = typer.Option(False, "--demo")):
    """Пересмотреть замечание по исходным материалам проверки."""
    try:
        _dispute(audit_id, req_id, comment, demo)
    except Exception as exc:
        _error(exc)
        raise typer.Exit(1)


@app.command()
def config():
    """Настроить подключения к моделям."""
    try:
        _configure()
    except SelectionCancelled:
        return
    except Exception as exc:
        _error(exc)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
