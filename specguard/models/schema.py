"""Pydantic models and data contracts for SpecGuard."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ViolationSeverity(str, Enum):
    CONFIRMED = "CONFIRMED"      # Подтвержденное противоречие ТЗ
    SUSPECTED = "SUSPECTED"      # Подозрение / неоднозначность


class RequirementStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"  # Требование реализовано в изменении
    VIOLATED = "VIOLATED"        # Обнаружено нарушение требования
    NOT_AFFECTED = "NOT_AFFECTED"# Не затронуто данным изменением


class SpecRequirement(BaseModel):
    """Единичное требование, извлеченное из ТЗ."""
    req_id: str
    section: str = ""
    title: str = ""
    description: str
    conditions: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    raw_text: str = ""
    source_file: str = ""


class Evidence(BaseModel):
    """Фактическое доказательство вывода."""
    file_path: str
    revision: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    code_quote: str = ""
    spec_quote: str = ""


class Finding(BaseModel):
    """Результат анализа конкретного требования."""
    req_id: str
    title: str
    status: RequirementStatus
    severity: Optional[ViolationSeverity] = None
    description: str
    evidence: List[Evidence] = Field(default_factory=list)
    counter_evidence: List[Evidence] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class FileChange(BaseModel):
    """Изменение конкретного файла в PR/коммите."""
    filename: str
    status: str  # "added", "modified", "deleted", "renamed"
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    patch: Optional[str] = None
    is_truncated: bool = False


class GitContext(BaseModel):
    """Контекст проверяемого изменения из Git/MCP."""
    target_repo: str
    target_ref: str                   # PR # (e.g. "PR#42") or Commit SHA
    base_sha: Optional[str] = None    # Исходная ревизия (или 1-й родитель merge-коммита)
    head_sha: str                     # Проверяемая ревизия
    is_merge_commit: bool = False
    first_parent_sha: Optional[str] = None
    files: List[FileChange] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    is_partial: bool = False


class AuditReport(BaseModel):
    """Итоговый отчет проверки соответствия."""
    audit_id: str
    timestamp: str
    target_repo: str
    target_ref: str
    base_parent_sha: Optional[str] = None
    user_summary: str
    language: str = "ru"
    findings: List[Finding] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    missing_data: List[str] = Field(default_factory=list)
    is_partial: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DisputeRequest(BaseModel):
    """Запрос на пересмотр спорного замечания."""
    audit_id: str
    finding_req_id: str
    user_comment: str
