"""Command Line Interface for SpecGuard (Module 4)."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from specguard.models.schema import (
    AuditReport,
    DisputeRequest,
    RequirementStatus,
    ViolationSeverity,
)
from specguard.agent.orchestrator import SpecGuardOrchestrator
from specguard.agent.llm_client import MockLLMClient, OpenAILLMClient
from specguard.rag.embeddings import (
    MockEmbeddingClient,
    OpenAIEmbeddingClient,
)
from specguard.rag.service import RAGService
from specguard.mcp.github_client import MockGitClient, GitHubMCPClient
from specguard.cli.history import HistoryStore
from specguard.cli.config import load_config


app = typer.Typer(
    help="SpecGuard: CLI-агент аудита PR и коммитов по техническому заданию."
)

console = Console()
history_store = HistoryStore()


def create_embedding_client(config):
    """Create a real or mock embedding client from configuration."""

    if config.embedding.base_url and config.embedding.model:
        console.print("[green]Используется настроенный Embedding API.[/green]")
        return OpenAIEmbeddingClient(
            base_url=config.embedding.base_url,
            model_name=config.embedding.model,
            api_key=config.embedding.api_key or None,
        )

    console.print("[yellow]Embedding API не настроен. Используется MockEmbeddingClient.[/yellow]")
    return MockEmbeddingClient()


def create_llm_client(config):
    """Create a real or mock LLM client from configuration."""

    if config.llm.base_url and config.llm.model:
        console.print("[green]Используется настроенный LLM API.[/green]")
        return OpenAILLMClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            api_key=config.llm.api_key or None,
        )

    console.print("[yellow]LLM API не настроен. Используется MockLLMClient.[/yellow]")
    return MockLLMClient()


def _display_report(report: AuditReport):
    """Display an audit report in the terminal."""

    console.print()

    state = (
        "[yellow]Частичный аудит[/yellow]"
        if report.is_partial
        else "[green]Полный аудит[/green]"
    )

    console.print(
        Panel(
            f"ID аудита: [bold]{report.audit_id}[/bold]\n"
            f"Репозиторий: {report.target_repo}\n"
            f"Цель: {report.target_ref}\n"
            f"Базовая ревизия: {report.base_parent_sha or 'Недоступна'}\n"
            f"Язык: {report.language}\n"
            f"Статус: {state}\n"
            f"Пояснение: {report.user_summary}",
            title="Отчёт SpecGuard",
        )
    )

    table = Table(title="Результаты проверки требований")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Требование")
    table.add_column("Статус")
    table.add_column("Серьёзность")
    table.add_column("Доказательства")

    for finding in report.findings:
        if finding.status == RequirementStatus.IMPLEMENTED:
            color = "green"
        elif finding.status == RequirementStatus.VIOLATED:
            color = "red"
        elif finding.status == RequirementStatus.INSUFFICIENT_DATA:
            color = "yellow"
        else:
            color = "cyan"

        severity = (
            finding.severity.value
            if finding.severity is not None
            else "—"
        )

        evidence_text = "Нет подтверждённых цитат"

        if finding.evidence:
            evidence_text = "\n".join(
                f"{ev.file_path}:{ev.line_start or '?'}\n{ev.code_quote}"
                for ev in finding.evidence
            )

        table.add_row(
            finding.req_id,
            finding.title,
            f"[{color}]{finding.status.value}[/{color}]",
            severity,
            evidence_text,
        )

    console.print(table)

    if report.limitations:
        console.print("\n[bold yellow]Ограничения аудита:[/bold yellow]")
        for limitation in report.limitations:
            console.print(f" - {limitation}")

    if report.missing_data:
        console.print("\n[bold yellow]Недостающие данные:[/bold yellow]")
        for item in report.missing_data:
            console.print(f" - {item}")


@app.command()
def audit(
    spec: str = typer.Option(
        ...,
        "--spec",
        "-s",
        help="Путь к файлу технического задания.",
    ),
    repo: str = typer.Option(
        ...,
        "--repo",
        "-r",
        help="GitHub-репозиторий в формате owner/repo.",
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Номер PR или SHA коммита.",
    ),
    summary: str = typer.Option(
        "",
        "--summary",
        "-m",
        help="Краткое пояснение пользователя.",
    ),
    lang: str = typer.Option(
        "ru",
        "--lang",
        "-l",
        help="Язык отчёта: ru или en.",
    ),
    use_mock_git: bool = typer.Option(
        True,
        "--mock-git/--live-git",
        help="Использовать тестовый или настоящий GitHub-клиент.",
    ),
):
    """Run an audit of a pull request or commit."""

    if lang.lower() not in ("ru", "en"):
        console.print("[red]Ошибка: язык должен быть ru или en.[/red]")
        raise typer.Exit(1)

    if not Path(spec).is_file():
        console.print(f"[red]Файл ТЗ не найден: {spec}[/red]")
        raise typer.Exit(1)

    config = load_config()

    embedding_client = None
    llm_client = None
    git_client = None

    try:
        console.print("[bold cyan]Запуск аудита SpecGuard...[/bold cyan]")
        console.print(f"ТЗ: [green]{spec}[/green]")
        console.print(f"Цель: [green]{repo}@{target}[/green]")

        # 1. Create independent API clients.
        embedding_client = create_embedding_client(config)
        llm_client = create_llm_client(config)

        # 2. Parse the specification and build its RAG index.
        rag_service = RAGService(embedding_client=embedding_client)

        console.print("[cyan]Извлечение требований из ТЗ...[/cyan]")

        spec_index, snapshot, parse_warnings = rag_service.index_document(spec)

        console.print(
            f"[green]Извлечено требований: {len(snapshot.requirements)}[/green]"
        )
        console.print(f"ID снимка ТЗ: {snapshot.snapshot_id}")

        for warning in parse_warnings:
            console.print(f"[yellow]Предупреждение: {warning}[/yellow]")

        # 3. Select GitHub client.
        if use_mock_git:
            git_client = MockGitClient()
        else:
            git_client = GitHubMCPClient()

        # 4. Run the audit.
        orchestrator = SpecGuardOrchestrator(
            git_client=git_client,
            spec_index=spec_index,
            llm_client=llm_client,
            language=lang,
        )

        report = orchestrator.run_audit(
            repo=repo,
            target_ref=target,
            user_summary=summary,
            limitations=parse_warnings,
            language=lang,
        )

        # 5. Save the report.
        history_store.save(report)

        console.print(
            f"\n[green]Аудит сохранён. ID: {report.audit_id}[/green]"
        )

        # 6. Display the result.
        _display_report(report)

    except Exception as exc:
        console.print(f"\n[bold red]Ошибка аудита: {exc}[/bold red]")
        raise typer.Exit(1)

    finally:
        for client in (git_client, llm_client, embedding_client):
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception:
                    pass


@app.command("history")
def show_history():
    """Show the list of saved audits."""

    reports = history_store.list()

    if not reports:
        console.print("[yellow]История аудитов пока пуста.[/yellow]")
        return

    table = Table(title="История аудитов SpecGuard")
    table.add_column("ID", style="cyan")
    table.add_column("Дата")
    table.add_column("Репозиторий")
    table.add_column("Цель")
    table.add_column("Найдено замечаний")
    table.add_column("Частичный аудит")

    for report in reports:
        violations = sum(
            1
            for finding in report.findings
            if finding.status == RequirementStatus.VIOLATED
        )

        table.add_row(
            report.audit_id,
            report.timestamp,
            report.target_repo,
            report.target_ref,
            str(violations),
            "Да" if report.is_partial else "Нет",
        )

    console.print(table)


@app.command()
def show(
    audit_id: str = typer.Argument(
        ...,
        help="ID аудита из истории.",
    ),
):
    """Display a previously saved audit report."""

    report = history_store.get(audit_id)

    if report is None:
        console.print(f"[red]Аудит {audit_id} не найден.[/red]")
        raise typer.Exit(1)

    _display_report(report)


@app.command()
def dispute(
    audit_id: str = typer.Argument(
        ...,
        help="ID исходного аудита.",
    ),
    req_id: str = typer.Option(
        ...,
        "--req",
        "-r",
        help="ID оспариваемого требования.",
    ),
    comment: str = typer.Option(
        ...,
        "--comment",
        "-c",
        help="Объяснение пользователя.",
    ),
):
    """Re-evaluate a finding using the original saved audit materials."""

    report = history_store.get(audit_id)

    if report is None:
        console.print(f"[red]Аудит {audit_id} не найден.[/red]")
        raise typer.Exit(1)

    finding = next(
        (
            item
            for item in report.findings
            if item.req_id.upper() == req_id.upper()
        ),
        None,
    )

    if finding is None:
        console.print(
            f"[red]Требование {req_id} не найдено в аудите {audit_id}.[/red]"
        )
        raise typer.Exit(1)

    if not report.saved_requirements or report.saved_git_context is None:
        console.print(
            "[red]В исходном отчёте отсутствуют сохранённые материалы. "
            "Повторный анализ невозможен.[/red]"
        )
        raise typer.Exit(1)

    config = load_config()
    llm_client = create_llm_client(config)

    try:
        request = DisputeRequest(
            audit_id=audit_id,
            finding_req_id=req_id,
            user_comment=comment,
        )

        orchestrator = SpecGuardOrchestrator(
            llm_client=llm_client,
            language=report.language,
        )

        console.print(
            Panel(
                f"Аудит: {audit_id}\n"
                f"Требование: {req_id}\n"
                f"Исходный статус: {finding.status.value}\n"
                f"Комментарий: {comment}",
                title="Повторное рассмотрение",
            )
        )

        # This method uses the saved specification and Git context.
        # It does not fetch the repository or specification again.
        response = orchestrator.review_dispute(
            dispute=request,
            saved_report=report,
            language=report.language,
        )

        # Save the dispute result separately, preserving the original report.
        dispute_dir = history_store.directory / "disputes"
        dispute_dir.mkdir(parents=True, exist_ok=True)

        result_file = dispute_dir / f"{audit_id}_{req_id}.json"
        result_file.write_text(
            response.model_dump_json(indent=2),
            encoding="utf-8",
        )

        console.print(
            Panel(
                f"Результат: [bold]{response.dispute_status.value}[/bold]\n"
                f"Исходный статус: {response.original_status.value}\n"
                f"Новый статус: {response.updated_finding.status.value}\n"
                f"Объяснение: {response.reasoning}",
                title="Результат пересмотра",
            )
        )

        console.print(
            f"[green]Результат пересмотра сохранён: {result_file}[/green]"
        )

    except Exception as exc:
        console.print(f"[bold red]Ошибка пересмотра: {exc}[/bold red]")
        raise typer.Exit(1)

    finally:
        if hasattr(llm_client, "close"):
            llm_client.close()


if __name__ == "__main__":
    app()
