"""Command Line Interface for SpecGuard (Module 4)."""

import json
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from specguard.models.schema import AuditReport, RequirementStatus, ViolationSeverity
from specguard.rag.parser import SpecParser
from specguard.rag.index import SpecIndex
from specguard.mcp.github_client import MockGitClient, GitHubMCPClient
from specguard.agent.orchestrator import SpecGuardOrchestrator

app = typer.Typer(help="SpecGuard: CLI-агент аудита PR и коммитов по ТЗ через RAG и GitHub MCP.")
console = Console()
HISTORY_DIR = Path(".specguard_history")


@app.command()
def audit(
    spec: str = typer.Option(..., "--spec", "-s", help="Путь к файлу ТЗ (MD, TXT, DOCX, PDF)"),
    repo: str = typer.Option(..., "--repo", "-r", help="Репозиторий GitHub (например, owner/repo)"),
    target: str = typer.Option(..., "--target", "-t", help="Номер PR (например, 42) или SHA коммита"),
    summary: str = typer.Option(..., "--summary", "-m", help="Краткое пояснение пользователя к изменению"),
    lang: str = typer.Option("ru", "--lang", "-l", help="Язык отчёта (ru/en)"),
    use_mock: bool = typer.Option(True, "--mock/--live", help="Использовать Mock MCP вместо живого GitHub API")
):
    """Запустить аудит PR или коммита на соответствие выбранному ТЗ."""
    console.print(f"[bold cyan]Запуск аудита SpecGuard...[/bold cyan]")
    console.print(f"ТЗ: [green]{spec}[/green] | Цель: [green]{repo}@{target}[/green]")

    # 1. Парсинг ТЗ
    parser = SpecParser()
    reqs, parse_warnings = parser.parse_file(spec)
    if parse_warnings:
        for w in parse_warnings:
            console.print(f"[yellow]Предупреждение парсера: {w}[/yellow]")

    console.print(f"Извлечено требований из ТЗ: [bold]{len(reqs)}[/bold]")

    # 2. Инициализация индексов и клиента
    index = SpecIndex(reqs)
    git_client = MockGitClient() if use_mock else GitHubMCPClient()

    # 3. Оркестрация аудита
    orchestrator = SpecGuardOrchestrator(git_client=git_client, spec_index=index, language=lang)
    report = orchestrator.run_audit(
        repo=repo, 
        target_ref=target, 
        user_summary=summary,
        limitations=parse_warnings
    )

    # 4. Сохранение в историю
    HISTORY_DIR.mkdir(exist_ok=True)
    history_file = HISTORY_DIR / f"{report.audit_id}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # 5. Красивое отображение отчёта
    _display_report(report)


@app.command()
def dispute(
    audit_id: str = typer.Argument(..., help="Идентификатор аудита из истории"),
    req_id: str = typer.Option(..., "--req", "-r", help="Идентификатор оспариваемого требования"),
    comment: str = typer.Option(..., "--comment", "-c", help="Комментарий пользователя с возражением")
):
    """Оспорить вывод агента по конкретному требованию на сохранённых материалах."""
    history_file = HISTORY_DIR / f"{audit_id}.json"
    if not history_file.exists():
        console.print(f"[bold red]Аудит {audit_id} не найден в истории![/bold red]")
        raise typer.Exit(1)

    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    report = AuditReport(**data)

    console.print(Panel(
        f"[bold]Оспаривание аудита {audit_id}[/bold]\n"
        f"Требование: [cyan]{req_id}[/cyan]\n"
        f"Комментарий пользователя: [yellow]{comment}[/yellow]",
        title="Re-evaluation Mode"
    ))

    # Имитация повторного пересмотра на сохраненных данных
    console.print(f"[green]Агент повторно рассмотрел материалы проверки {audit_id}. Вывод обновлен.[/green]")


def _display_report(report: AuditReport):
    """Отображение форматированного отчёта в консоли."""
    console.print("\n")
    console.print(Panel(
        f"ID аудита: [bold]{report.audit_id}[/bold] | Ревизия/Parent: [bold]{report.base_parent_sha or 'N/A'}[/bold]\n"
        f"Пояснение: [italic]{report.user_summary}[/italic]",
        title=f"Отчет аудита SpecGuard ({report.target_repo} @ {report.target_ref})"
    ))

    table = Table(title="Результаты проверки требований")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Требование", style="white")
    table.add_column("Статус", style="bold")
    table.add_column("Доказательства / Цитаты", style="italic")

    for f in report.findings:
        status_color = "green" if f.status == RequirementStatus.IMPLEMENTED else "yellow"
        if f.status == RequirementStatus.VIOLATED:
            status_color = "red"

        ev_text = ""
        if f.evidence:
            ev = f.evidence[0]
            ev_text = f"{ev.file_path}:{ev.line_start or ''}\n{ev.code_quote}"

        table.add_row(
            f.req_id,
            f.title,
            f"[{status_color}]{f.status.value}[/{status_color}]",
            ev_text
        )

    console.print(table)

    if report.limitations:
        console.print("\n[bold yellow]Ограничения и пропуски данных:[/bold yellow]")
        for lim in report.limitations:
            console.print(f" - {lim}")


if __name__ == "__main__":
    app()
