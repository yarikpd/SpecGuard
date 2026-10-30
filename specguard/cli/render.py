"""Complete reports as literal Rich text, safe for untrusted source quotes."""
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
    parts = [Panel(Text(header), title="SpecGuard", border_style="#725d65")]
    snapshots = report.metadata.get("spec_snapshots", [])
    if snapshots:
        parts.append(Panel(Text("\n".join(f"{item['path']} · {item['snapshot_id']}" for item in snapshots)),
                           title=tr("Документы ТЗ", "Specification documents")))
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
