"""History storage for SpecGuard audits."""

import json
from pathlib import Path
from typing import Optional

from specguard.models.schema import AuditReport


class HistoryStore:
    """Stores and retrieves audit reports as JSON files."""

    def __init__(self, directory: str = ".specguard_history"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, report: AuditReport) -> Path:
        """Save an audit report."""
        file_path = self.directory / f"{report.audit_id}.json"
        temp_path = file_path.with_suffix(".tmp")

        temp_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temp_path.replace(file_path)

        return file_path

    def get(self, audit_id: str) -> Optional[AuditReport]:
        """Get an audit report by ID."""
        file_path = self.directory / f"{audit_id}.json"

        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text(encoding="utf-8"))
        return AuditReport.model_validate(data)

    def list(self) -> list[AuditReport]:
        """Return all saved audit reports."""
        reports = []

        for file_path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                reports.append(AuditReport.model_validate(data))
            except (json.JSONDecodeError, ValueError):
                continue

        return reports
