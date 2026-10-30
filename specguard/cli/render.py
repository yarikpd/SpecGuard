"""Complete reports as literal Rich text, safe for untrusted source quotes."""
import re
from rich.style import Style
from specguard.mcp.github_client import parse_repo
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render_report(report):
    en = report.language == "en"
    def tr(ru, english): return english if en else ru
    context = report.saved_git_context
    header = '\n'.join([
        f"ID: {report.audit_id}", f"{report.target_repo} · {report.target_ref}",
        f"Base: {report.base_parent_sha or '—'}", f"Head: {context.head_sha if context else '—'}",
        tr("Частичный результат", "Partial result") if report.is_partial else tr("Анализ завершён", "Analysis complete"),
        f"Mode: {report.metadata.get('mode', 'unspecified')}",
        f"{tr('Пояснение', 'Summary')}: {report.user_summary}",
        f"{tr('Снимок ТЗ', 'Specification snapshot')}: {report.spec_snapshot_id or '—'}",
    ])
    header_text = Text(header)
    number = context.pr_number if context else None
    if number is None:
        ref = context.target_ref if context else report.target_ref
        match = re.fullmatch(r"(?:(?:refs/)?pull/|pr\s*#?|#)?(\d+)(?:/head|/merge)?", ref.strip(), re.I)
        if match and (not ref.isdigit() or len(ref) <= 7):
            number = int(match.group(1))
    if number:
        owner, repo = parse_repo(report.target_repo)
        url = f"https://github.com/{owner}/{repo}/pull/{number}"
        title = context.pr_title if context else None
        label = f"PR #{number}" + (f" · {title}" if title else "")
        link = Text(label, style=Style(color="#c6afb6", underline=True, link=url,
                                      meta={"@click": f"app.open_pr('{url}')"}))
        header_text = Text.assemble(link, "\n", header_text)
    parts = [Panel(header_text, title="SpecGuard", border_style="#725d65")]
    snapshots = report.metadata.get("spec_snapshots", [])
    if snapshots:
        parts.append(Panel(Text("\n".join(f"{item['path']} · {item['snapshot_id']}" for item in snapshots)),
                           title=tr("Документы ТЗ", "Specification documents")))
    scope_lines = [tr(
        "Наблюдения для ручной оценки. Отсутствие найденной связи с ТЗ само по себе не является нарушением. При неполном извлечении ТЗ вывод ограничен доступным текстом.",
        "Observations for human review. No identified specification link is not itself a violation. Incomplete extraction limits this review to available text.")]
    for observation in report.scope_observations:
        scope_lines.extend(["", observation.title, observation.description])
        for ev in observation.evidence:
            scope_lines.extend([f"{ev.file_path}:{ev.line_start}–{ev.line_end} @ {ev.revision}", ev.code_quote])
    if not report.scope_observations:
        if report.metadata.get("scope_review") == "completed":
            scope_lines.append(tr("Такие изменения не выявлены при этом анализе.", "No such changes were identified in this analysis."))
        else:
            scope_lines.append(tr("Отдельная проверка не выполнена или не завершена.", "The separate review was not run or did not complete."))
    parts.append(Panel(Text('\n'.join(scope_lines)),
                       title=tr("Изменения без найденной связи с ТЗ", "Changes without an identified specification link"),
                       border_style="#c6afb6"))
    table = Table(expand=True)
    for column in ["ID", tr("Требование", "Requirement"), tr("Статус", "Status"), tr("Уверенность", "Confidence")]:
        table.add_column(column)
    for finding in report.findings:
        table.add_row(Text(finding.req_id), Text(finding.title), Text(finding.status.value),
                      Text(finding.severity.value if finding.severity else "—"))
    parts.append(table)
    for finding in report.findings:
        lines = [finding.description]
        for label, evidence in [(tr("Доказательства", "Evidence"), finding.evidence),
                                (tr("Контрдоказательства", "Counter-evidence"), finding.counter_evidence)]:
            for ev in evidence:
                lines.extend([f"\n{label}: {ev.file_path}:{ev.line_start or '?'}–{ev.line_end or '?'} @ {ev.revision}",
                              tr("Код:", "Code:"), ev.code_quote, tr("ТЗ:", "Specification:"), ev.spec_quote])
        if not finding.evidence:
            lines.append(tr("Нет подтверждённых цитат кода", "No verified code quotes"))
        lines.extend(f"{tr('Ограничение', 'Limitation')}: {value}" for value in finding.limitations)
        parts.append(Panel(Text('\n'.join(lines)), title=Text(f"{finding.req_id} · {finding.title}")))
    for label, values in [(tr("Ограничения", "Limitations"), report.limitations),
                          (tr("Недостающие данные", "Missing data"), report.missing_data)]:
        if values:
            parts.append(Panel(Text('\n'.join(values)), title=label, border_style="#e2bd89"))
    if context:
        parts.append(Panel(Text('\n'.join(
            f"{f.filename} · {f.status}" + (" · partial" if f.is_truncated or f.patch is None else "")
            for f in context.files)), title=tr("Область проверки", "Scope")))
    return Group(*parts)
