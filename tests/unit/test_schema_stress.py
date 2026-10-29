"""Empirical stress and property test suite for specguard.models.schema.

Challenger 1 stress harness verifying:
1. Deeply nested models (large AuditReports with hundreds of Findings & thousands of Evidence objects)
2. Default factory isolation (verifying no shared mutable state between distinct instances)
3. Extreme boundary values (empty strings, huge string payloads, Unicode/Cyrillic characters, None vs omitted)
4. Validation rejection on illegal enum values and type mismatches
5. Pydantic v2 JSON roundtrip fidelity across all types and edge cases
6. Model contract consistency (DisputeResponse dispute_status, negative line numbers, reversed line ranges, severity consistency)
"""

import json
import time
from datetime import datetime
from typing import Any, Dict
import pytest
from pydantic import ValidationError

from specguard.models.schema import (
    AuditReport,
    DisputeRequest,
    DisputeResponse,
    DisputeStatus,
    Evidence,
    FileChange,
    Finding,
    GitContext,
    RequirementStatus,
    SpecRequirement,
    ViolationSeverity,
)


# ============================================================================
# 1. Deeply Nested Models & Scalability Stress
# ============================================================================

class TestDeepNestingAndScaleStress:
    """Stress tests on deeply nested models and large collection sizes."""

    def test_massive_nested_audit_report(self):
        """Stress test: 500 Findings, each with 5 Evidence items (total 2500 Evidence items)."""
        findings = []
        for i in range(500):
            ev_list = [
                Evidence(
                    file_path=f"src/module_{i}/file_{j}.py",
                    revision=f"commit_{i:04d}_{j:04d}",
                    line_start=j * 10 + 1,
                    line_end=j * 10 + 20,
                    code_quote=f"code snippet for finding {i} evidence {j}",
                    spec_quote=f"spec quote for finding {i} evidence {j}",
                )
                for j in range(5)
            ]
            findings.append(
                Finding(
                    req_id=f"REQ-{i:04d}",
                    title=f"Requirement {i} validation check",
                    status=RequirementStatus.VIOLATED if i % 2 == 0 else RequirementStatus.IMPLEMENTED,
                    severity=ViolationSeverity.CONFIRMED if i % 2 == 0 else None,
                    description=f"Automated evaluation description for requirement {i}.",
                    evidence=ev_list,
                    counter_evidence=[],
                    limitations=[f"limitation_{i}"],
                )
            )

        report = AuditReport(
            audit_id="audit-stress-scale-500",
            target_repo="specguard/stress-repo",
            target_ref="refs/pull/999/merge",
            user_summary="Massive scale stress test report",
            language="ru",
            findings=findings,
            limitations=["Large diff processed"],
            missing_data=[],
            is_partial=False,
            metadata={"finding_count": len(findings)},
        )

        assert len(report.findings) == 500
        total_evidence = sum(len(f.evidence) for f in report.findings)
        assert total_evidence == 2500

        # Measure serialization and deserialization
        start_dump = time.perf_counter()
        json_str = report.model_dump_json()
        dump_duration = time.perf_counter() - start_dump
        assert dump_duration < 2.0  # Must serialize within 2 seconds

        start_load = time.perf_counter()
        restored = AuditReport.model_validate_json(json_str)
        load_duration = time.perf_counter() - start_load
        assert load_duration < 2.0  # Must parse within 2 seconds

        assert len(restored.findings) == 500
        assert sum(len(f.evidence) for f in restored.findings) == 2500
        assert restored == report


# ============================================================================
# 2. Default Factory Isolation Stress
# ============================================================================

class TestDefaultFactoryIsolation:
    """Verify that mutable default collections are isolated across instances."""

    def test_audit_report_default_factories_isolated(self):
        r1 = AuditReport(
            audit_id="r1",
            target_repo="repo/1",
            target_ref="ref1",
            user_summary="First report",
        )
        r2 = AuditReport(
            audit_id="r2",
            target_repo="repo/2",
            target_ref="ref2",
            user_summary="Second report",
        )

        # Confirm distinct references
        assert r1.findings is not r2.findings
        assert r1.limitations is not r2.limitations
        assert r1.missing_data is not r2.missing_data
        assert r1.metadata is not r2.metadata

        # Mutate r1 collections
        f = Finding(
            req_id="FR-001",
            title="T",
            status=RequirementStatus.IMPLEMENTED,
            description="D",
        )
        r1.findings.append(f)
        r1.limitations.append("only r1 limit")
        r1.missing_data.append("only r1 missing")
        r1.metadata["custom"] = "value"

        # Assert r2 remains completely unaffected
        assert len(r2.findings) == 0
        assert len(r2.limitations) == 0
        assert len(r2.missing_data) == 0
        assert len(r2.metadata) == 0
        assert "custom" not in r2.metadata

    def test_finding_default_factories_isolated(self):
        f1 = Finding(
            req_id="F1",
            title="T1",
            status=RequirementStatus.IMPLEMENTED,
            description="D1",
        )
        f2 = Finding(
            req_id="F2",
            title="T2",
            status=RequirementStatus.IMPLEMENTED,
            description="D2",
        )

        assert f1.evidence is not f2.evidence
        assert f1.counter_evidence is not f2.counter_evidence
        assert f1.limitations is not f2.limitations

        f1.evidence.append(Evidence(file_path="a.py", revision="r1"))
        f1.counter_evidence.append(Evidence(file_path="b.py", revision="r2"))
        f1.limitations.append("f1 limitation")

        assert len(f2.evidence) == 0
        assert len(f2.counter_evidence) == 0
        assert len(f2.limitations) == 0

    def test_git_context_default_factories_isolated(self):
        g1 = GitContext(target_repo="repo/a", target_ref="ref1", head_sha="sha1")
        g2 = GitContext(target_repo="repo/b", target_ref="ref2", head_sha="sha2")

        assert g1.files is not g2.files
        assert g1.limitations is not g2.limitations

        g1.files.append(FileChange(filename="foo.py", status="added"))
        g1.limitations.append("git limitation")

        assert len(g2.files) == 0
        assert len(g2.limitations) == 0

    def test_spec_requirement_default_factories_isolated(self):
        s1 = SpecRequirement(req_id="S1", description="Desc 1")
        s2 = SpecRequirement(req_id="S2", description="Desc 2")

        assert s1.conditions is not s2.conditions
        assert s1.exceptions is not s2.exceptions
        assert s1.related_req_ids is not s2.related_req_ids

        s1.conditions.append("cond1")
        s1.exceptions.append("exc1")
        s1.related_req_ids.append("rel1")

        assert len(s2.conditions) == 0
        assert len(s2.exceptions) == 0
        assert len(s2.related_req_ids) == 0

    def test_audit_report_timestamp_dynamically_generated(self):
        """Verify timestamp default factory creates new timestamps rather than static cached value."""
        r1 = AuditReport(
            audit_id="r1",
            target_repo="repo/1",
            target_ref="ref1",
            user_summary="Report 1",
        )
        time.sleep(0.005)
        r2 = AuditReport(
            audit_id="r2",
            target_repo="repo/2",
            target_ref="ref2",
            user_summary="Report 2",
        )
        # Parse ISO timestamps
        dt1 = datetime.fromisoformat(r1.timestamp)
        dt2 = datetime.fromisoformat(r2.timestamp)
        assert dt2 >= dt1


# ============================================================================
# 3. Extreme Boundary Values & Unicode / Cyrillic Fidelity
# ============================================================================

class TestExtremeBoundariesAndUnicode:
    """Stress testing on extreme strings, Unicode/Cyrillic scripts, and boundary numbers."""

    def test_huge_string_payload_roundtrip(self):
        """1 megabyte diff patch and huge code quote."""
        huge_patch = "A" * (1024 * 1024)  # 1MB
        huge_quote = "B" * (512 * 1024)   # 512KB

        fc = FileChange(
            filename="large_data.blob",
            status="modified",
            patch=huge_patch,
            is_truncated=False,
        )
        ev = Evidence(
            file_path="large_data.blob",
            revision="deadbeef",
            code_quote=huge_quote,
        )
        finding = Finding(
            req_id="PERF-001",
            title="Large payload handling",
            status=RequirementStatus.IMPLEMENTED,
            description="Testing handling of MB-scale string fields",
            evidence=[ev],
        )
        report = AuditReport(
            audit_id="audit-huge-payload",
            target_repo="specguard/repo",
            target_ref="refs/heads/main",
            user_summary="Huge payload test",
            findings=[finding],
        )

        json_str = report.model_dump_json()
        restored = AuditReport.model_validate_json(json_str)

        assert len(restored.findings[0].evidence[0].code_quote) == 512 * 1024
        assert restored.findings[0].evidence[0].code_quote == huge_quote

    def test_multilingual_unicode_and_emojis(self):
        """Cyrillic, Arabic, Chinese, Japanese, emojis, ZWJ sequences, and math symbols."""
        cyrillic_text = "Аудит изменений: обнаружено несоответствие разделу ТЗ 3.1"
        arabic_text = "تدقيق مواصفات الكود البرمجي"
        cjk_text = "仕様準拠監査システム - 漏洞检测"
        emoji_text = "🚀 ⚠️ 👨‍👩‍👧‍👦 🔍 🛡️ ❌ ✅"
        special_chars = "<script>alert('xss')</script> \t \n \r \" ' \\ / \0"

        ev = Evidence(
            file_path="путь/к/файлу/модуль.py",
            revision="ревизия_123",
            line_start=1,
            line_end=10,
            code_quote=f"code: {emoji_text} // {special_chars}",
            spec_quote=f"spec: {cyrillic_text} | {arabic_text} | {cjk_text}",
        )
        finding = Finding(
            req_id="ТРЕБ-001",
            title=f"Проверка {cyrillic_text}",
            status=RequirementStatus.VIOLATED,
            severity=ViolationSeverity.CONFIRMED,
            description=f"{cyrillic_text}\n{arabic_text}\n{cjk_text}\n{emoji_text}",
            evidence=[ev],
            limitations=["Ограничение 1", "Ограничение 2"],
        )
        report = AuditReport(
            audit_id="аудит-001-рус",
            target_repo="репозиторий/проект",
            target_ref="ветка/фича-1",
            user_summary=f"Отчет аудита {emoji_text}",
            language="ru",
            findings=[finding],
            limitations=["Ограничений нет"],
            missing_data=["Недостающие данные отсутствуют"],
        )

        json_str = report.model_dump_json()
        restored = AuditReport.model_validate_json(json_str)

        assert restored == report
        assert restored.findings[0].req_id == "ТРЕБ-001"
        assert cyrillic_text in restored.findings[0].description
        assert arabic_text in restored.findings[0].description
        assert cjk_text in restored.findings[0].description
        assert emoji_text in restored.user_summary

    def test_extreme_integer_ranges_on_line_numbers(self):
        """Test line_start and line_end with 0, large integers, and None."""
        ev_zero = Evidence(
            file_path="src/a.py",
            revision="r1",
            line_start=0,
            line_end=0,
        )
        assert ev_zero.line_start == 0
        assert ev_zero.line_end == 0

        ev_large = Evidence(
            file_path="src/big.py",
            revision="r1",
            line_start=10**9,
            line_end=10**9 + 100,
        )
        assert ev_large.line_start == 10**9
        assert ev_large.line_end == 10**9 + 100

        ev_none = Evidence(
            file_path="src/a.py",
            revision="r1",
            line_start=None,
            line_end=None,
        )
        assert ev_none.line_start is None
        assert ev_none.line_end is None

    def test_empty_string_fields_allowed(self):
        """Verify models accept empty strings for string fields."""
        report = AuditReport(
            audit_id="",
            timestamp="",
            target_repo="",
            target_ref="",
            user_summary="",
            language="",
        )
        assert report.audit_id == ""
        assert report.user_summary == ""

        req = DisputeRequest(
            audit_id="",
            finding_req_id="",
            user_comment="",
        )
        assert req.user_comment == ""


# ============================================================================
# 4. Validation Rejection Stress
# ============================================================================

class TestValidationRejectionStress:
    """Challenge validation rejection on malformed inputs and illegal types."""

    def test_invalid_status_enum_rejection(self):
        with pytest.raises(ValidationError) as exc_info:
            Finding(
                req_id="FR-001",
                title="T",
                status="COMPLETED",  # Invalid enum value
                description="D",
            )
        assert "status" in str(exc_info.value)

    def test_invalid_severity_enum_rejection(self):
        with pytest.raises(ValidationError) as exc_info:
            Finding(
                req_id="FR-001",
                title="T",
                status=RequirementStatus.VIOLATED,
                severity="CRITICAL",  # Invalid enum value
                description="D",
            )
        assert "severity" in str(exc_info.value)

    def test_none_not_allowed_for_required_fields(self):
        with pytest.raises(ValidationError):
            Evidence(file_path=None, revision="rev")  # file_path cannot be None

        with pytest.raises(ValidationError):
            Evidence(file_path="a.py", revision=None)  # revision cannot be None

        with pytest.raises(ValidationError):
            AuditReport(
                audit_id=None,  # audit_id cannot be None
                target_repo="r",
                target_ref="ref",
            )

        with pytest.raises(ValidationError):
            Finding(
                req_id="FR-001",
                title="T",
                status=None,  # status cannot be None
                description="D",
            )

    def test_invalid_types_for_nested_collections(self):
        with pytest.raises(ValidationError):
            AuditReport(
                audit_id="a",
                target_repo="r",
                target_ref="ref",
                findings=12345,  # int instead of list
            )

        with pytest.raises(ValidationError):
            AuditReport(
                audit_id="a",
                target_repo="r",
                target_ref="ref",
                findings=["not a finding object"],  # string instead of Finding
            )

        with pytest.raises(ValidationError):
            AuditReport(
                audit_id="a",
                target_repo="r",
                target_ref="ref",
                metadata=["not", "a", "dict"],  # list instead of dict
            )

    def test_invalid_types_for_boolean_fields(self):
        with pytest.raises(ValidationError):
            AuditReport(
                audit_id="a",
                target_repo="r",
                target_ref="ref",
                is_partial={"key": "val"},  # dict cannot coerce to bool
            )

        with pytest.raises(ValidationError):
            FileChange(
                filename="a.py",
                status="added",
                is_truncated=[1, 2, 3],  # list cannot coerce to bool
            )

    def test_invalid_types_for_line_numbers(self):
        with pytest.raises(ValidationError):
            Evidence(
                file_path="a.py",
                revision="r1",
                line_start="invalid_int_string",
            )


# ============================================================================
# 5. Contract Cohesion & Potential Architectural Inconsistencies
# ============================================================================

class TestContractInconsistencies:
    """Test potential gaps in schema definitions and contracts."""

    def test_dispute_response_accepts_dispute_status_enum(self):
        """Verify DisputeResponse can accept DisputeStatus enum values without error."""
        f = Finding(
            req_id="FR-001",
            title="T",
            status=RequirementStatus.VIOLATED,
            severity=ViolationSeverity.CONFIRMED,
            description="D",
        )
        resp = DisputeResponse(
            audit_id="a1",
            finding_req_id="FR-001",
            original_status=RequirementStatus.VIOLATED,
            original_severity=ViolationSeverity.CONFIRMED,
            dispute_status=DisputeStatus.OVERTURNED,  # Passing Enum directly
            updated_finding=f,
            reasoning="Reason",
        )
        assert resp.dispute_status == "OVERTURNED"
        assert resp.dispute_status == DisputeStatus.OVERTURNED

    def test_dispute_response_arbitrary_string_allowed_by_schema(self):
        """Observation: dispute_status is typed as str, so arbitrary non-enum strings are permitted.
        
        This matches PROJECT.md interface contract line 72 ('dispute_status: str'),
        which allows custom statuses like 'REVISED', but leaves DisputeStatus enum un-enforced.
        """
        f = Finding(
            req_id="FR-001",
            title="T",
            status=RequirementStatus.IMPLEMENTED,
            description="D",
        )
        resp = DisputeResponse(
            audit_id="a1",
            finding_req_id="FR-001",
            original_status=RequirementStatus.IMPLEMENTED,
            dispute_status="CUSTOM_UNKNOWN_STATUS",
            updated_finding=f,
            reasoning="Custom reasoning",
        )
        assert resp.dispute_status == "CUSTOM_UNKNOWN_STATUS"

    def test_evidence_negative_and_reversed_line_numbers_behavior(self):
        """Observation: Evidence does not restrict negative line numbers or line_end < line_start.
        
        PROJECT.md defines line_start: Optional[int], line_end: Optional[int].
        The models do not add extra validator constraints, allowing maximum compatibility.
        """
        ev_neg = Evidence(file_path="a.py", revision="r1", line_start=-5, line_end=-1)
        assert ev_neg.line_start == -5

        ev_reversed = Evidence(file_path="a.py", revision="r1", line_start=100, line_end=20)
        assert ev_reversed.line_start == 100
        assert ev_reversed.line_end == 20

    def test_violated_finding_allows_none_severity(self):
        """Observation: Finding allows status=VIOLATED with severity=None.
        
        PROJECT.md specifies severity: Optional[ViolationSeverity], allowing None for all statuses.
        """
        finding = Finding(
            req_id="FR-001",
            title="T",
            status=RequirementStatus.VIOLATED,
            severity=None,
            description="D",
        )
        assert finding.status == RequirementStatus.VIOLATED
        assert finding.severity is None
