from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ViolationSeverity(str, Enum):
    CONFIRMED = "CONFIRMED"
    SUSPECTED = "SUSPECTED"


class RequirementStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    VIOLATED = "VIOLATED"
    NOT_AFFECTED = "NOT_AFFECTED"


class SpecRequirement(BaseModel):
    req_id: str
    section: str = ""
    title: str = ""
    description: str
    conditions: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    raw_text: str = ""
    source_file: str = ""
    related_req_ids: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    file_path: str
    revision: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    code_quote: str = ""
    spec_quote: str = ""


class Finding(BaseModel):
    req_id: str
    title: str
    status: RequirementStatus
    severity: Optional[ViolationSeverity] = None
    description: str
    evidence: List[Evidence] = Field(default_factory=list)
    counter_evidence: List[Evidence] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class FileChange(BaseModel):
    filename: str
    status: str
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    patch: Optional[str] = None
    is_truncated: bool = False


class GitContext(BaseModel):
    target_repo: str
    target_ref: str
    base_sha: Optional[str] = None
    head_sha: str
    is_merge_commit: bool = False
    first_parent_sha: Optional[str] = None
    files: List[FileChange] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    is_partial: bool = False


class AuditReport(BaseModel):
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
    audit_id: str
    finding_req_id: str
    user_comment: str
