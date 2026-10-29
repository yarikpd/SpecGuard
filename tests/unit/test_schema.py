"""Unit test suite for specguard.models.schema canonical contracts.

Validates:
- Enums: RequirementStatus, ViolationSeverity
- Models: Evidence, Finding, FileChange, GitContext, AuditReport, DisputeRequest, DisputeResponse, SpecRequirement
- Partial data flags (is_partial, limitations, missing_data, base_parent_sha)
- Merge commit parent tracking (first_parent_sha)
- Pydantic v2 JSON roundtrip serialization and deserialization
- Validation error handling on missing fields and invalid types
"""

import json
from typing import Any, Dict
import pytest
from pydantic import ValidationError

from specguard.models.schema import (
    AuditReport,
    DisputeRequest,
    DisputeResponse,
    Evidence,
    FileChange,
    Finding,
    GitContext,
    RequirementStatus,
    SpecRequirement,
    ViolationSeverity,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_evidence() -> Evidence:
    return Evidence(
        file_path="specguard/agent/orchestrator.py",
        revision="a1b2c3d4e5f6",
        line_start=45,
        line_end=60,
        code_quote="def run_audit(self, ...): ...",
        spec_quote="FR-004: Agent detects possible violations",
    )


@pytest.fixture
def sample_finding_violated(sample_evidence: Evidence) -> Finding:
    return Finding(
        req_id="FR-004",
        title="Missing violation check for merge commits",
        status=RequirementStatus.VIOLATED,
        severity=ViolationSeverity.CONFIRMED,
        description="The orchestrator fails to handle merge commit first parents.",
        evidence=[sample_evidence],
        counter_evidence=[],
        limitations=["Only git diff checked"],
    )


@pytest.fixture
def sample_audit_report(sample_finding_violated: Finding) -> AuditReport:
    return AuditReport(
        audit_id="audit-2026-001",
        timestamp="2026-10-29T15:30:00Z",
        target_repo="owner/specguard-repo",
        target_ref="refs/pull/42/merge",
        base_parent_sha="1a2b3c4d5e6f",
        user_summary="Added merge commit tracking and dispute procedure",
        language="ru",
        findings=[sample_finding_violated],
        limitations=["Diff truncated at 500 lines"],
        missing_data=["tests/fixtures/unreachable.py"],
        is_partial=True,
        metadata={"run_mode": "mock", "orchestrator_version": "0.1.0"},
    )


# ============================================================================
# 1. Enums
# ============================================================================

class TestEnums:
    """Tests for RequirementStatus and ViolationSeverity enums."""

    def test_requirement_status_members_and_values(self):
        assert RequirementStatus.IMPLEMENTED == "IMPLEMENTED"
        assert RequirementStatus.VIOLATED == "VIOLATED"
        assert RequirementStatus.NOT_AFFECTED == "NOT_AFFECTED"
        assert set(RequirementStatus) == {
            RequirementStatus.IMPLEMENTED,
            RequirementStatus.VIOLATED,
            RequirementStatus.NOT_AFFECTED,
        }

    def test_requirement_status_from_string(self):
        assert RequirementStatus("IMPLEMENTED") is RequirementStatus.IMPLEMENTED
        assert RequirementStatus("VIOLATED") is RequirementStatus.VIOLATED
        assert RequirementStatus("NOT_AFFECTED") is RequirementStatus.NOT_AFFECTED

    def test_requirement_status_invalid_value(self):
        with pytest.raises(ValueError):
            RequirementStatus("UNKNOWN_STATUS")

    def test_violation_severity_members_and_values(self):
        assert ViolationSeverity.CONFIRMED == "CONFIRMED"
        assert ViolationSeverity.SUSPECTED == "SUSPECTED"
        assert set(ViolationSeverity) == {
            ViolationSeverity.CONFIRMED,
            ViolationSeverity.SUSPECTED,
        }

    def test_violation_severity_from_string(self):
        assert ViolationSeverity("CONFIRMED") is ViolationSeverity.CONFIRMED
        assert ViolationSeverity("SUSPECTED") is ViolationSeverity.SUSPECTED

    def test_violation_severity_invalid_value(self):
        with pytest.raises(ValueError):
            ViolationSeverity("CRITICAL")


# ============================================================================
# 2. Evidence Model
# ============================================================================

class TestEvidenceModel:
    """Tests for Evidence creation and field validation."""

    def test_evidence_minimal_creation(self):
        ev = Evidence(file_path="main.py", revision="abc1234")
        assert ev.file_path == "main.py"
        assert ev.revision == "abc1234"
        assert ev.line_start is None
        assert ev.line_end is None
        assert ev.code_quote == ""
        assert ev.spec_quote == ""

    def test_evidence_full_creation(self):
        ev = Evidence(
            file_path="auth/service.py",
            revision="def5678",
            line_start=10,
            line_end=25,
            code_quote="def verify_token(): ...",
            spec_quote="NFR-002: Token verification",
        )
        assert ev.file_path == "auth/service.py"
        assert ev.revision == "def5678"
        assert ev.line_start == 10
        assert ev.line_end == 25
        assert ev.code_quote == "def verify_token(): ..."
        assert ev.spec_quote == "NFR-002: Token verification"

    def test_evidence_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Evidence(file_path="main.py")  # missing revision
        with pytest.raises(ValidationError):
            Evidence(revision="abc1234")  # missing file_path

    def test_evidence_invalid_line_number_type(self):
        with pytest.raises(ValidationError):
            Evidence(file_path="a.py", revision="123", line_start="not_an_int")


# ============================================================================
# 3. Finding Model
# ============================================================================

class TestFindingModel:
    """Tests for Finding model across various statuses and severities."""

    def test_finding_implemented(self, sample_evidence: Evidence):
        finding = Finding(
            req_id="FR-001",
            title="User login implemented",
            status=RequirementStatus.IMPLEMENTED,
            severity=None,
            description="Login method added and verified against spec.",
            evidence=[sample_evidence],
        )
        assert finding.status == RequirementStatus.IMPLEMENTED
        assert finding.severity is None
        assert len(finding.evidence) == 1
        assert finding.counter_evidence == []
        assert finding.limitations == []

    def test_finding_violated_confirmed(self, sample_evidence: Evidence):
        finding = Finding(
            req_id="FR-002",
            title="Password length validation omitted",
            status=RequirementStatus.VIOLATED,
            severity=ViolationSeverity.CONFIRMED,
            description="Password check is completely absent in auth controller.",
            evidence=[sample_evidence],
        )
        assert finding.status == RequirementStatus.VIOLATED
        assert finding.severity == ViolationSeverity.CONFIRMED

    def test_finding_violated_suspected(self, sample_evidence: Evidence):
        finding = Finding(
            req_id="FR-003",
            title="Possible rate limit omission",
            status=RequirementStatus.VIOLATED,
            severity=ViolationSeverity.SUSPECTED,
            description="Rate limiter configuration not found in PR diff.",
            evidence=[sample_evidence],
            counter_evidence=[
                Evidence(file_path="config.py", revision="rev1", code_quote="RATE_LIMIT = 100")
            ],
            limitations=["External rate limiter in NGINX might handle this."],
        )
        assert finding.status == RequirementStatus.VIOLATED
        assert finding.severity == ViolationSeverity.SUSPECTED
        assert len(finding.counter_evidence) == 1
        assert len(finding.limitations) == 1

    def test_finding_not_affected(self):
        finding = Finding(
            req_id="FR-010",
            title="Database snapshots untouched",
            status=RequirementStatus.NOT_AFFECTED,
            severity=None,
            description="PR only modifies documentation and comments.",
        )
        assert finding.status == RequirementStatus.NOT_AFFECTED
        assert finding.severity is None
        assert finding.evidence == []

    def test_finding_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Finding(req_id="FR-001", title="Title", description="Desc")  # missing status

    def test_finding_invalid_status_enum(self):
        with pytest.raises(ValidationError):
            Finding(
                req_id="FR-001",
                title="T",
                status="INVALID_STATUS",
                description="D",
            )

    def test_finding_invalid_severity_enum(self):
        with pytest.raises(ValidationError):
            Finding(
                req_id="FR-001",
                title="T",
                status=RequirementStatus.VIOLATED,
                severity="HIGH",
                description="D",
            )


# ============================================================================
# 4. GitContext and FileChange Models
# ============================================================================

class TestGitModels:
    """Tests for FileChange and GitContext models."""

    def test_file_change_minimal(self):
        fc = FileChange(filename="src/app.py", status="modified")
        assert fc.filename == "src/app.py"
        assert fc.status == "modified"
        assert fc.old_path is None
        assert fc.new_path is None
        assert fc.patch is None
        assert fc.is_truncated is False

    def test_file_change_truncated(self):
        fc = FileChange(
            filename="data/huge.json",
            status="added",
            patch="@@ -0,0 +1,10000 @@ ... [TRUNCATED]",
            is_truncated=True,
        )
        assert fc.is_truncated is True
        assert "TRUNCATED" in fc.patch

    def test_file_change_missing_required(self):
        with pytest.raises(ValidationError):
            FileChange(filename="test.py")  # missing status

    def test_git_context_standard_commit(self):
        ctx = GitContext(
            target_repo="specguard/core",
            target_ref="refs/heads/feature",
            base_sha="base001",
            head_sha="head002",
            is_merge_commit=False,
            first_parent_sha=None,
            is_partial=False,
        )
        assert ctx.is_merge_commit is False
        assert ctx.first_parent_sha is None
        assert ctx.files == []
        assert ctx.limitations == []
        assert ctx.is_partial is False

    def test_git_context_merge_commit_with_first_parent(self):
        ctx = GitContext(
            target_repo="specguard/core",
            target_ref="refs/pull/15/merge",
            base_sha="master_sha_111",
            head_sha="merge_sha_222",
            is_merge_commit=True,
            first_parent_sha="parent1_sha_333",
        )
        assert ctx.is_merge_commit is True
        assert ctx.first_parent_sha == "parent1_sha_333"

    def test_git_context_partial_and_truncated(self):
        fc = FileChange(filename="bundle.min.js", status="modified", is_truncated=True)
        ctx = GitContext(
            target_repo="specguard/core",
            target_ref="refs/pull/99/head",
            head_sha="head999",
            files=[fc],
            limitations=["Diff exceeded 5000 lines limit"],
            is_partial=True,
        )
        assert ctx.is_partial is True
        assert len(ctx.limitations) == 1
        assert ctx.files[0].is_truncated is True

    def test_git_context_missing_required(self):
        with pytest.raises(ValidationError):
            GitContext(target_repo="owner/repo", target_ref="ref")  # missing head_sha


# ============================================================================
# 5. AuditReport Model
# ============================================================================

class TestAuditReportModel:
    """Tests for AuditReport creation, defaults, and edge cases."""

    def test_audit_report_defaults(self):
        report = AuditReport(
            audit_id="audit-100",
            timestamp="2026-10-29T10:00:00Z",
            target_repo="owner/repo",
            target_ref="refs/heads/main",
            user_summary="Minor doc fix",
        )
        assert report.language == "ru"
        assert report.findings == []
        assert report.limitations == []
        assert report.missing_data == []
        assert report.is_partial is False
        assert report.metadata == {}
        assert report.base_parent_sha is None

    def test_audit_report_merge_parent_sha_recorded(self, sample_audit_report: AuditReport):
        assert sample_audit_report.base_parent_sha == "1a2b3c4d5e6f"

    def test_audit_report_partial_flags(self, sample_audit_report: AuditReport):
        assert sample_audit_report.is_partial is True
        assert len(sample_audit_report.limitations) == 1
        assert len(sample_audit_report.missing_data) == 1

    def test_audit_report_missing_required(self):
        with pytest.raises(ValidationError):
            AuditReport(
                audit_id="audit-100",
                timestamp="2026-10-29T10:00:00Z",
                target_repo="owner/repo",
                # missing target_ref and user_summary
            )

    def test_audit_report_invalid_nested_finding(self):
        with pytest.raises(ValidationError):
            AuditReport(
                audit_id="audit-100",
                timestamp="2026-10-29T10:00:00Z",
                target_repo="owner/repo",
                target_ref="refs/heads/main",
                user_summary="Summary",
                findings=[{"req_id": "FR-1"}],  # invalid finding payload
            )


# ============================================================================
# 6. DisputeRequest and DisputeResponse Models
# ============================================================================

class TestDisputeModels:
    """Tests for DisputeRequest and DisputeResponse models."""

    def test_dispute_request_valid(self):
        req = DisputeRequest(
            audit_id="audit-2026-001",
            finding_req_id="FR-004",
            user_comment="The check is executed in middleware/auth.py lines 10-15.",
        )
        assert req.audit_id == "audit-2026-001"
        assert req.finding_req_id == "FR-004"
        assert "middleware/auth.py" in req.user_comment

    def test_dispute_request_missing_required(self):
        with pytest.raises(ValidationError):
            DisputeRequest(audit_id="audit-1", finding_req_id="FR-1")  # missing user_comment

    def test_dispute_response_overturned(self, sample_finding_violated: Finding):
        updated_finding = sample_finding_violated.model_copy(
            update={
                "status": RequirementStatus.IMPLEMENTED,
                "severity": None,
                "description": "Re-evaluated: middleware confirms requirement satisfaction.",
            }
        )
        resp = DisputeResponse(
            audit_id="audit-2026-001",
            finding_req_id="FR-004",
            original_status=RequirementStatus.VIOLATED,
            original_severity=ViolationSeverity.CONFIRMED,
            dispute_status="OVERTURNED",
            updated_finding=updated_finding,
            reasoning="Examined saved middleware artifact; verified compliant.",
            timestamp="2026-10-29T16:00:00Z",
        )
        assert resp.audit_id == "audit-2026-001"
        assert resp.original_status == RequirementStatus.VIOLATED
        assert resp.original_severity == ViolationSeverity.CONFIRMED
        assert resp.dispute_status == "OVERTURNED"
        assert resp.updated_finding.status == RequirementStatus.IMPLEMENTED
        assert resp.updated_finding.severity is None

    def test_dispute_response_upheld(self, sample_finding_violated: Finding):
        resp = DisputeResponse(
            audit_id="audit-2026-001",
            finding_req_id="FR-004",
            original_status=RequirementStatus.VIOLATED,
            original_severity=ViolationSeverity.CONFIRMED,
            dispute_status="UPHELD",
            updated_finding=sample_finding_violated,
            reasoning="Middleware check does not satisfy exception handling requirement.",
            timestamp="2026-10-29T16:05:00Z",
        )
        assert resp.dispute_status == "UPHELD"
        assert resp.updated_finding.status == RequirementStatus.VIOLATED

    def test_dispute_response_optional_severity(self, sample_finding_violated: Finding):
        resp = DisputeResponse(
            audit_id="audit-2026-001",
            finding_req_id="FR-001",
            original_status=RequirementStatus.IMPLEMENTED,
            original_severity=None,
            dispute_status="REVISED",
            updated_finding=sample_finding_violated,
            reasoning="User noted missing edge case.",
            timestamp="2026-10-29T16:10:00Z",
        )
        assert resp.original_severity is None

    def test_dispute_response_missing_required(self):
        with pytest.raises(ValidationError):
            DisputeResponse(
                audit_id="audit-1",
                finding_req_id="FR-1",
                original_status=RequirementStatus.VIOLATED,
                # missing dispute_status, updated_finding, reasoning, timestamp
            )


# ============================================================================
# 7. SpecRequirement Model
# ============================================================================

class TestSpecRequirementModel:
    """Tests for SpecRequirement schema."""

    def test_spec_requirement_creation(self):
        req = SpecRequirement(
            req_id="FR-001",
            section="1. Authentication",
            title="Email Login",
            description="Users must be able to log in with email and password.",
            conditions=["Active account"],
            exceptions=["Blocked users"],
            raw_text="FR-001: Email login description",
            source_file="spec.md",
            related_req_ids=["FR-002"],
        )
        assert req.req_id == "FR-001"
        assert req.conditions == ["Active account"]
        assert req.exceptions == ["Blocked users"]
        assert req.related_req_ids == ["FR-002"]

    def test_spec_requirement_defaults(self):
        req = SpecRequirement(req_id="FR-002", description="Password length >= 8")
        assert req.section == ""
        assert req.title == ""
        assert req.conditions == []
        assert req.exceptions == []
        assert req.raw_text == ""
        assert req.source_file == ""
        assert req.related_req_ids == []


# ============================================================================
# 8. Pydantic v2 Serialization & Deserialization (Roundtrip)
# ============================================================================

class TestSerializationRoundtrip:
    """Tests for model_dump_json and model_validate_json roundtrip integrity."""

    def test_audit_report_json_roundtrip(self, sample_audit_report: AuditReport):
        json_data = sample_audit_report.model_dump_json()
        assert isinstance(json_data, str)

        parsed = json.loads(json_data)
        assert parsed["audit_id"] == "audit-2026-001"
        assert parsed["is_partial"] is True
        assert parsed["findings"][0]["status"] == "VIOLATED"
        assert parsed["findings"][0]["severity"] == "CONFIRMED"

        restored = AuditReport.model_validate_json(json_data)
        assert restored == sample_audit_report
        assert isinstance(restored.findings[0].status, RequirementStatus)
        assert isinstance(restored.findings[0].severity, ViolationSeverity)

    def test_dispute_response_json_roundtrip(self, sample_finding_violated: Finding):
        resp = DisputeResponse(
            audit_id="audit-2026-001",
            finding_req_id="FR-004",
            original_status=RequirementStatus.VIOLATED,
            original_severity=ViolationSeverity.CONFIRMED,
            dispute_status="OVERTURNED",
            updated_finding=sample_finding_violated,
            reasoning="Examined saved materials.",
            timestamp="2026-10-29T16:00:00Z",
        )
        json_str = resp.model_dump_json()
        restored = DisputeResponse.model_validate_json(json_str)
        assert restored == resp
        assert restored.original_status is RequirementStatus.VIOLATED
        assert restored.original_severity is ViolationSeverity.CONFIRMED

    def test_dispute_request_json_roundtrip(self):
        req = DisputeRequest(
            audit_id="audit-001",
            finding_req_id="FR-002",
            user_comment="Notice tests pass in CI.",
        )
        json_str = req.model_dump_json()
        restored = DisputeRequest.model_validate_json(json_str)
        assert restored == req

    def test_git_context_json_roundtrip(self):
        ctx = GitContext(
            target_repo="specguard/core",
            target_ref="refs/pull/1/merge",
            head_sha="head123",
            first_parent_sha="parent123",
            is_merge_commit=True,
            files=[FileChange(filename="a.py", status="added", is_truncated=False)],
            limitations=["None"],
            is_partial=False,
        )
        json_str = ctx.model_dump_json()
        restored = GitContext.model_validate_json(json_str)
        assert restored == ctx

    def test_cyrillic_text_serialization_fidelity(self):
        report = AuditReport(
            audit_id="audit-cyrillic",
            timestamp="2026-10-29T12:00:00Z",
            target_repo="owner/репозиторий",
            target_ref="refs/heads/ветка",
            user_summary="Краткое описание проверки на русском языке",
            language="ru",
            limitations=["Ограничение по размеру диффа"],
            missing_data=["Файл не найден"],
        )
        json_str = report.model_dump_json()
        restored = AuditReport.model_validate_json(json_str)
        assert restored == report
        assert "Краткое описание" in restored.user_summary
        assert "Ограничение по размеру" in restored.limitations[0]


# ============================================================================
# 9. Generic Validation Failures
# ============================================================================

class TestValidationFailures:
    """Tests for edge cases and type mismatch validation errors."""

    @pytest.mark.parametrize(
        "model_cls",
        [
            Evidence,
            Finding,
            FileChange,
            GitContext,
            AuditReport,
            DisputeRequest,
            DisputeResponse,
            SpecRequirement,
        ],
    )
    def test_empty_payload_raises_validation_error(self, model_cls):
        with pytest.raises(ValidationError):
            model_cls.model_validate({})

    def test_invalid_type_for_boolean_field(self):
        with pytest.raises(ValidationError):
            AuditReport.model_validate({
                "audit_id": "a",
                "timestamp": "t",
                "target_repo": "r",
                "target_ref": "ref",
                "user_summary": "s",
                "is_partial": ["not", "a", "boolean"],
            })

    def test_invalid_type_for_findings_list(self):
        with pytest.raises(ValidationError):
            AuditReport.model_validate({
                "audit_id": "a",
                "timestamp": "t",
                "target_repo": "r",
                "target_ref": "ref",
                "user_summary": "s",
                "findings": "should_be_a_list",
            })
