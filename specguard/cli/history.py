"""Persistent, append-only audit and dispute history."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional
from uuid import uuid4

from specguard.models.schema import AuditReport, DisputeRequest, DisputeResponse


class HistoryStore:
    """Keep original audit materials and every subsequent review separately."""

    def __init__(self, directory: str | Path = ".specguard_history"):
        self.directory = Path(directory)

    @staticmethod
    def _validate_id(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value):
            raise ValueError("Некорректный ID аудита")
        return value

    @staticmethod
    def _write_new(path: Path, data: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents concurrent runs from replacing history.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(data)
        return path

    def save(self, report: AuditReport) -> Path:
        """Save a run; assign a fresh ID when the core repeats a deterministic ID."""
        original_id = self._validate_id(report.audit_id)
        while True:
            file_path = self.directory / f"{report.audit_id}.json"
            try:
                return self._write_new(file_path, report.model_dump_json(indent=2))
            except FileExistsError:
                report.audit_id = f"{original_id[:120]}_{uuid4().hex}"

    def get(self, audit_id: str) -> Optional[AuditReport]:
        file_path = self.directory / f"{self._validate_id(audit_id)}.json"
        if not file_path.exists():
            return None
        return AuditReport.model_validate_json(file_path.read_text(encoding="utf-8"))

    def list(self) -> list[AuditReport]:
        reports = []
        for file_path in sorted(self.directory.glob("*.json")):
            try:
                reports.append(AuditReport.model_validate_json(file_path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
        return sorted(reports, key=lambda report: report.timestamp, reverse=True)

    def save_dispute(self, request: DisputeRequest, response: DisputeResponse) -> Path:
        """Persist the comment and result without replacing previous reviews."""
        audit_id = self._validate_id(request.audit_id)
        if response.audit_id != audit_id or response.finding_req_id != request.finding_req_id:
            raise ValueError("Результат пересмотра не соответствует запросу")
        if self.get(audit_id) is None:
            raise ValueError("Исходный аудит не найден")
        data = {"request": request.model_dump(mode="json"), "response": response.model_dump(mode="json")}
        return self._write_new(
            self.directory / "disputes" / audit_id / f"{uuid4().hex}.json",
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    def list_disputes(self, audit_id: str) -> list[dict]:
        folder = self.directory / "disputes" / self._validate_id(audit_id)
        results = []
        for path in folder.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            DisputeRequest.model_validate(data["request"])
            DisputeResponse.model_validate(data["response"])
            results.append(data)
        return sorted(results, key=lambda item: item["response"]["timestamp"])
