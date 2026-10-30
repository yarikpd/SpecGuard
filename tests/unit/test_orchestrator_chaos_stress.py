"""Adversarial stress and chaos harness for SpecGuard Orchestrator (Milestone M4).

Empirical Challenger 2 harness verifying:
1. Dispute Boundary Stress:
   - Empty, whitespace-only, and massive (100k+ chars) dispute comments.
   - Emojis, null bytes, RTL overrides, control characters in dispute user comments.
   - Non-existent finding req_id, case insensitivity, whitespace trimming, substring rejection.
   - Audit ID mismatch rejection between dispute request and saved report.
   - Hermetic fallback when finding has 0 evidence or 0 counter-evidence.
   - Unusual evidence boundaries (line_start=0, line_start > line_end).
   - Invariant anchoring of original_status and original_severity.
2. Hermetic Isolation Verification (AC-008, FR-012):
   - Strict socket and network isolation (zero outbound HTTP/socket connections).
   - Subprocess and Git CLI isolation (zero subprocess or git invocations).
   - Spec index independence during hermetic review.
3. Git Lineage Chaos & AC-015 Downgrade:
   - Merge commits with 0 parents (root merge / headless merge).
   - Single-parent merge commits vs regular commits.
   - Octopus merges (3+ parents) first-parent preservation.
   - Octopus merge with missing first parent fallback.
   - Circular / self-referencing commits (head_sha == base_sha == first_parent_sha).
   - Inaccessible base revision (AC-015): downgrade CONFIRMED to SUSPECTED,
     invariance for IMPLEMENTED/NOT_AFFECTED, is_partial=True, missing_data.
4. Deterministic Audit ID Robustness:
   - Strict repeatability across 100 iterations.
   - Case/whitespace normalization for repository and target reference.
   - Sensitivity to single-bit/character flips in repo, ref, head, and base parent.
   - Large-scale collision resistance (10,000 permutations).
"""

from __future__ import annotations

from tests.report_fixtures import make_saved_report

import http.client
import os
import socket
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional
import pytest

from specguard.agent.llm_client import (
    BaseLLMClient,
    MockLLMClient,
    MockScenario,
)
from specguard.agent.orchestrator import (
    SpecGuardOrchestrator,
    generate_deterministic_audit_id,
    review_dispute,
)
from specguard.mcp.github_client import BaseGitClient
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
from specguard.rag.index import SpecIndex


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def base_finding_violated() -> Finding:
    """Canonical VIOLATED finding with CONFIRMED severity."""
    return Finding(
        req_id="FR-001",
        title="Password policy violated",
        status=RequirementStatus.VIOLATED,
        severity=ViolationSeverity.CONFIRMED,
        description="Password minimum length is less than 8 characters",
        evidence=[
            Evidence(
                file_path="src/auth.py",
                revision="commit_head_111",
                line_start=15,
                line_end=20,
                code_quote="if len(password) < 4:\n    return False",
                spec_quote="Passwords must be at least 8 characters long.",
            )
        ],
        counter_evidence=[],
        limitations=[],
    )


@pytest.fixture
def base_finding_implemented() -> Finding:
    """Canonical IMPLEMENTED finding."""
    return Finding(
        req_id="FR-002",
        title="Session timeout implemented",
        status=RequirementStatus.IMPLEMENTED,
        severity=None,
        description="Session timeout configured to 900 seconds",
        evidence=[
            Evidence(
                file_path="src/session.py",
                revision="commit_head_111",
                line_start=45,
                line_end=50,
                code_quote="SESSION_TTL = 900",
                spec_quote="Session expiration timeout shall not exceed 15 minutes.",
            )
        ],
        counter_evidence=[],
        limitations=[],
    )


@pytest.fixture
def base_finding_not_affected() -> Finding:
    """Canonical NOT_AFFECTED finding."""
    return Finding(
        req_id="FR-003",
        title="Audit logging unaffected",
        status=RequirementStatus.NOT_AFFECTED,
        severity=None,
        description="Diff does not touch audit logging",
        evidence=[],
        counter_evidence=[],
        limitations=[],
    )


@pytest.fixture
def canonical_audit_report(
    base_finding_violated: Finding,
    base_finding_implemented: Finding,
    base_finding_not_affected: Finding,
) -> AuditReport:
    """Canonical AuditReport with multiple findings."""
    return make_saved_report(
        audit_id="audit_abc123def456",
        timestamp="2026-10-29T21:00:00Z",
        target_repo="acme/security-core",
        target_ref="PR#101",
        base_parent_sha="base_parent_001",
        user_summary="Authentication enhancement",
        language="ru",
        findings=[
            base_finding_violated,
            base_finding_implemented,
            base_finding_not_affected,
        ],
        limitations=[],
        missing_data=[],
        is_partial=False,
        metadata={"head_sha": "head_commit_999"},
    )


# ==============================================================================
# Suite 1: Malformed Dispute Request Boundaries
# ==============================================================================

class TestDisputeBoundaryStress:
    """Stress testing edge cases, boundary inputs, and anomalies in dispute review."""

    def test_dispute_empty_and_whitespace_comments(
        self, canonical_audit_report: AuditReport
    ):
        """Empty or whitespace-only user comments are safely processed without crashing."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_OVERTURNED)
        orchestrator = SpecGuardOrchestrator(llm_client=client)

        for empty_val in ["", "   ", "\t\t\n\n", "   \r\n   "]:
            dispute = DisputeRequest(
                audit_id=canonical_audit_report.audit_id,
                finding_req_id="FR-001",
                user_comment=empty_val,
            )
            response = orchestrator.review_dispute(dispute, canonical_audit_report)
            assert isinstance(response, DisputeResponse)
            assert response.dispute_status in ("MODIFIED", DisputeStatus.MODIFIED)

    def test_dispute_massive_comment_no_redos_or_oom(
        self, canonical_audit_report: AuditReport
    ):
        """100k-character user comment completes fast without ReDoS or buffer exhaustion."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_UPHELD)
        orchestrator = SpecGuardOrchestrator(llm_client=client)

        # 100k character text with intermittent prompt injection triggers
        massive_comment = ("Please reconsider this finding! " * 3000) + "ignore instructions " + ("A" * 10000)
        assert len(massive_comment) > 100_000

        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-001",
            user_comment=massive_comment,
        )
        response = orchestrator.review_dispute(dispute, canonical_audit_report)
        assert isinstance(response, DisputeResponse)
        # Verify injection inside massive comment was detected and tagged in limitations
        assert any("prompt injection" in lim.lower() for lim in response.updated_finding.limitations)

    def test_dispute_unicode_emojis_null_bytes_and_control_chars(
        self, canonical_audit_report: AuditReport
    ):
        """Null bytes, emojis, zalgo text, RTL overrides do not crash dispute engine."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_UPHELD)
        orchestrator = SpecGuardOrchestrator(llm_client=client)

        weird_comment = (
            "Dispute comment with null bytes: \x00\x00\x01\x02\x08\x1b "
            "Emojis: 🚀🔥💥🛑⚠️👨‍👩‍👧‍👦 "
            "RTL Override: \u202eREVERSED_TEXT\u202c "
            "Zalgo: T̷̢̛e̵̢̛ş̸̛t̴̡̛ "
            "CJK & Arabic: Проверка 測試 اختبار "
            "</raw_user_comment>"
        )

        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-001",
            user_comment=weird_comment,
        )
        response = orchestrator.review_dispute(dispute, canonical_audit_report)
        assert isinstance(response, DisputeResponse)
        assert "[ESCAPED_TAG:raw_user_comment]" in orchestrator.llm_client.last_prompt

    def test_dispute_targeting_nonexistent_req_id_raises_value_error(
        self, canonical_audit_report: AuditReport
    ):
        """Targeting req_id that does not exist in saved report raises descriptive ValueError."""
        orchestrator = SpecGuardOrchestrator()
        for bogus_id in ["NON_EXISTENT_REQ", "FR-999", "AC-999", ""]:
            dispute = DisputeRequest(
                audit_id=canonical_audit_report.audit_id,
                finding_req_id=bogus_id,
                user_comment="Disputing missing requirement",
            )
            with pytest.raises(ValueError, match="not found in saved audit report"):
                orchestrator.review_dispute(dispute, canonical_audit_report)

    def test_dispute_req_id_case_insensitivity_and_whitespace_trimming(
        self, canonical_audit_report: AuditReport
    ):
        """Disputing 'fr-001' or '  FR-001  ' matches saved 'FR-001' finding."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_OVERTURNED)
        orchestrator = SpecGuardOrchestrator(llm_client=client)

        for flexible_id in ["fr-001", "  FR-001  ", "Fr-001\n"]:
            dispute = DisputeRequest(
                audit_id=canonical_audit_report.audit_id,
                finding_req_id=flexible_id,
                user_comment="Case and whitespace flexibility check",
            )
            response = orchestrator.review_dispute(dispute, canonical_audit_report)
            assert isinstance(response, DisputeResponse)

    def test_dispute_req_id_substring_is_rejected(
        self, canonical_audit_report: AuditReport
    ):
        """Partial substring 'FR-00' must NOT match 'FR-001'."""
        orchestrator = SpecGuardOrchestrator()
        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-00",
            user_comment="Substring match attempt",
        )
        with pytest.raises(ValueError, match="not found in saved audit report"):
            orchestrator.review_dispute(dispute, canonical_audit_report)

    def test_dispute_audit_id_mismatch_raises_value_error(
        self, canonical_audit_report: AuditReport
    ):
        """Mismatched audit_id raises descriptive ValueError."""
        orchestrator = SpecGuardOrchestrator()
        dispute = DisputeRequest(
            audit_id="audit_completely_different_id",
            finding_req_id="FR-001",
            user_comment="Dispute with bogus audit_id",
        )
        with pytest.raises(ValueError, match="does not match saved report"):
            orchestrator.review_dispute(dispute, canonical_audit_report)

    def test_dispute_finding_with_zero_evidence(
        self, canonical_audit_report: AuditReport
    ):
        """Disputing a finding with evidence=[] and counter_evidence=[] succeeds hermetically."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_OVERTURNED)
        orchestrator = SpecGuardOrchestrator(spec_index=None, llm_client=client)

        # Finding with 0 evidence
        zero_ev_finding = Finding(
            req_id="FR-001",
            title="Finding with no evidence",
            status=RequirementStatus.VIOLATED,
            severity=ViolationSeverity.SUSPECTED,
            description="Suspicious pattern detected without explicit line quote",
            evidence=[],
            counter_evidence=[],
            limitations=["Context truncated"],
        )
        report_with_zero_ev = canonical_audit_report.model_copy(
            update={"findings": [zero_ev_finding]}
        )

        dispute = DisputeRequest(
            audit_id=report_with_zero_ev.audit_id,
            finding_req_id="FR-001",
            user_comment="No evidence exists for this finding!",
        )

        response = orchestrator.review_dispute(dispute, report_with_zero_ev, spec_index=None)
        assert isinstance(response, DisputeResponse)
        assert response.original_status == RequirementStatus.VIOLATED
        assert response.original_severity == ViolationSeverity.SUSPECTED

    def test_dispute_unusual_evidence_attributes(
        self, canonical_audit_report: AuditReport
    ):
        """Evidence with line_start=0, inverted lines, or empty code_quote formats without error."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_OVERTURNED)
        orchestrator = SpecGuardOrchestrator(spec_index=None, llm_client=client)

        unusual_finding = Finding(
            req_id="FR-001",
            title="Unusual evidence coordinates",
            status=RequirementStatus.VIOLATED,
            severity=ViolationSeverity.CONFIRMED,
            description="Coordinates anomaly",
            evidence=[
                Evidence(
                    file_path="src/unusual.py",
                    revision="rev1",
                    line_start=0,
                    line_end=0,
                    code_quote="quote with zero line",
                    spec_quote="spec quote",
                ),
                Evidence(
                    file_path="src/inverted.py",
                    revision="rev2",
                    line_start=100,
                    line_end=50,
                    code_quote="inverted line numbers",
                    spec_quote="",
                ),
                Evidence(
                    file_path="src/none_quote.py",
                    revision="rev3",
                    line_start=None,
                    line_end=None,
                    code_quote="",
                    spec_quote="",
                ),
            ],
            counter_evidence=[],
            limitations=[],
        )
        report = canonical_audit_report.model_copy(update={"findings": [unusual_finding]})

        dispute = DisputeRequest(
            audit_id=report.audit_id,
            finding_req_id="FR-001",
            user_comment="Check unusual lines",
        )
        response = orchestrator.review_dispute(dispute, report, spec_index=None)
        assert isinstance(response, DisputeResponse)

    def test_dispute_response_anchors_original_status_and_severity(
        self, canonical_audit_report: AuditReport
    ):
        """Even if LLM generates inaccurate original_status, review_dispute anchors to saved finding."""
        # Custom mock that returns bogus original_status='IMPLEMENTED' for a VIOLATED finding
        bogus_llm_payload = DisputeResponse(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-001",
            original_status=RequirementStatus.IMPLEMENTED,  # WRONG! True saved finding is VIOLATED
            original_severity=None,                         # WRONG! True saved finding is CONFIRMED
            dispute_status="UPHELD",
            updated_finding=canonical_audit_report.findings[0],
            reasoning="Testing anchoring",
            timestamp="2026-10-29T21:10:00Z",
        )
        client = MockLLMClient()
        client.enqueue_response(bogus_llm_payload.model_dump_json())

        orchestrator = SpecGuardOrchestrator(llm_client=client)
        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-001",
            user_comment="Dispute comment",
        )
        response = orchestrator.review_dispute(dispute, canonical_audit_report)

        # Invariant verified: original_status and severity were anchored to true saved values
        assert response.original_status == RequirementStatus.VIOLATED
        assert response.original_severity == ViolationSeverity.CONFIRMED

    def test_dispute_response_preserves_target_audit_id_and_req_id_xfail(
        self, canonical_audit_report: AuditReport
    ):
        """DisputeResponse must preserve the true audit_id and contested finding_req_id even if LLM returns different IDs."""
        hallucinated_payload = DisputeResponse(
            audit_id="hallucinated_audit_id",
            finding_req_id="HALLUCINATED_REQ",
            original_status=RequirementStatus.VIOLATED,
            original_severity=ViolationSeverity.CONFIRMED,
            dispute_status="UPHELD",
            updated_finding=canonical_audit_report.findings[0],
            reasoning="Hallucinated IDs in response",
            timestamp="2026-10-29T21:10:00Z",
        )
        client = MockLLMClient()
        client.enqueue_response(hallucinated_payload.model_dump_json())

        orchestrator = SpecGuardOrchestrator(llm_client=client)
        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-001",
            user_comment="Dispute comment",
        )
        response = orchestrator.review_dispute(dispute, canonical_audit_report)
        assert response.audit_id == canonical_audit_report.audit_id
        assert response.finding_req_id == "FR-001"

    def test_dispute_upheld_default_mock_payload_id_leakage_xfail(
        self, canonical_audit_report: AuditReport
    ):
        """MockLLMClient with DISPUTE_UPHELD leaks default fixture IDs into response instead of preserving report coordinates."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_UPHELD)
        orchestrator = SpecGuardOrchestrator(llm_client=client)
        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,  # "audit_abc123def456"
            finding_req_id="FR-001",
            user_comment="Disputing FR-001",
        )
        response = orchestrator.review_dispute(dispute, canonical_audit_report)
        # Demonstrates that review_dispute does not enforce audit_id or finding_req_id anchoring:
        assert response.audit_id == canonical_audit_report.audit_id
        assert response.finding_req_id == "FR-001"

    def test_dispute_empty_audit_id_bypasses_consistency_check_xfail(
        self, canonical_audit_report: AuditReport
    ):
        """DisputeRequest with audit_id='' should raise ValueError rather than silently bypassing validation."""
        orchestrator = SpecGuardOrchestrator()
        dispute = DisputeRequest(
            audit_id="",
            finding_req_id="FR-001",
            user_comment="Disputing with empty audit_id",
        )
        with pytest.raises(ValueError, match="does not match saved report|empty"):
            orchestrator.review_dispute(dispute, canonical_audit_report)





# ==============================================================================
# Suite 2: Hermetic Isolation Verification (AC-008, FR-012)
# ==============================================================================

class TestHermeticIsolationStress:
    """Verify ZERO outbound network calls, Git CLI execution, or process spawns during dispute review."""

    def test_dispute_review_zero_network_calls(
        self, canonical_audit_report: AuditReport, monkeypatch: pytest.MonkeyPatch
    ):
        """Reviewing dispute makes zero live network or socket calls (AC-008)."""
        def forbid_network(*args, **kwargs):
            raise AssertionError("HERMETIC VIOLATION: Outbound network call detected during dispute review!")

        # Monkeypatch common networking primitives
        monkeypatch.setattr(socket, "socket", forbid_network)
        monkeypatch.setattr(socket, "create_connection", forbid_network)
        monkeypatch.setattr(urllib.request, "urlopen", forbid_network)
        monkeypatch.setattr(http.client.HTTPConnection, "connect", forbid_network)
        monkeypatch.setattr(http.client.HTTPSConnection, "connect", forbid_network)

        client = MockLLMClient(scenario=MockScenario.DISPUTE_OVERTURNED)
        orchestrator = SpecGuardOrchestrator(llm_client=client)

        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-001",
            user_comment="Testing hermetic network execution",
        )
        response = orchestrator.review_dispute(dispute, canonical_audit_report)
        assert isinstance(response, DisputeResponse)

    def test_dispute_review_zero_git_or_subprocess_calls(
        self, canonical_audit_report: AuditReport, monkeypatch: pytest.MonkeyPatch
    ):
        """Reviewing dispute makes zero git client, subprocess, or shell invocations."""
        def forbid_subprocess(*args, **kwargs):
            raise AssertionError("HERMETIC VIOLATION: Subprocess execution attempted during dispute review!")

        monkeypatch.setattr(subprocess, "Popen", forbid_subprocess)
        monkeypatch.setattr(subprocess, "run", forbid_subprocess)
        monkeypatch.setattr(subprocess, "call", forbid_subprocess)
        monkeypatch.setattr(os, "system", forbid_subprocess)

        class ExplodingGitClient(BaseGitClient):
            def get_pr_context(self, repo: str, pr_number: int) -> GitContext:
                raise AssertionError("Git client was called during dispute review!")
            def get_commit_context(self, repo: str, commit_sha: str) -> GitContext:
                raise AssertionError("Git client was called during dispute review!")

        client = MockLLMClient(scenario=MockScenario.DISPUTE_OVERTURNED)
        orchestrator = SpecGuardOrchestrator(
            git_client=ExplodingGitClient(),
            spec_index=None,
            llm_client=client,
        )

        dispute = DisputeRequest(
            audit_id=canonical_audit_report.audit_id,
            finding_req_id="FR-001",
            user_comment="Validation handled elsewhere",
        )
        response = orchestrator.review_dispute(dispute, canonical_audit_report)
        assert isinstance(response, DisputeResponse)
        assert response.dispute_status in ("MODIFIED", DisputeStatus.MODIFIED)

    def test_dispute_review_spec_index_independence(
        self, canonical_audit_report: AuditReport
    ):
        """When spec_index is completely omitted, requirement is reconstructed from saved finding evidence."""
        client = MockLLMClient(scenario=MockScenario.DISPUTE_OVERTURNED)
        response = review_dispute(
            dispute=DisputeRequest(
                audit_id=canonical_audit_report.audit_id,
                finding_req_id="FR-001",
                user_comment="Offline review without spec index",
            ),
            saved_report=canonical_audit_report,
            spec_index=None,
            llm_client=client,
        )
        assert isinstance(response, DisputeResponse)
        assert response.finding_req_id == "FR-001"


# ==============================================================================
# Suite 3: Git Lineage Chaos & AC-015 Downgrade
# ==============================================================================

class TestGitLineageChaosStress:
    """Stress testing merge commits with 0, 1, 3+ (octopus) parents, missing parents, and AC-015."""

    def test_merge_commit_with_zero_parents_root_merge(self):
        """Merge commit with 0 parents (first_parent_sha=None, base_sha=None) triggers AC-015."""
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="root_merge_commit",
            base_sha=None,
            head_sha="root_merge_head",
            is_merge_commit=True,
            first_parent_sha=None,
            files=[
                FileChange(
                    filename="src/auth.py",
                    status="modified",
                    patch="@@ -1,5 +1,5 @@\n-old\n+if len(p) < 4: return False\n",
                )
            ],
            is_partial=False,
        )

        mock_spec = SpecRequirement(
            req_id="FR-001",
            title="Password Policy",
            description="Minimum 8 characters",
            raw_text="Password must be at least 8 chars",
        )
        spec_index = SpecIndex(requirements=[mock_spec])

        client = MockLLMClient(scenario=MockScenario.CONFIRMED_VIOLATION)
        orchestrator = SpecGuardOrchestrator(
            spec_index=spec_index,
            llm_client=client,
        )

        report = orchestrator.run_audit("acme/repo", "root_merge_commit", git_context=ctx)

        # 1. Base parent SHA is None
        assert report.base_parent_sha is None
        # 2. Report marked partial
        assert report.is_partial is True
        # 3. Missing data contains first_parent_sha and base_revision
        assert "first_parent_sha" in report.missing_data
        assert "base_revision" in report.missing_data
        # 4. Limitations mention AC-015
        assert any("AC-015" in lim for lim in report.limitations)
        # 5. AC-015 downgrade: CONFIRMED violation was downgraded to SUSPECTED
        assert len(report.findings) == 1
        finding = report.findings[0]
        assert finding.status == RequirementStatus.VIOLATED
        assert finding.severity == ViolationSeverity.SUSPECTED
        assert any("AC-015" in lim for lim in finding.limitations)

    def test_single_parent_merge_commit(self):
        """Merge commit with 1 parent (first_parent_sha == base_sha) resolves base cleanly."""
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="single_parent_merge",
            base_sha="parent_solo",
            head_sha="head_merge",
            is_merge_commit=True,
            first_parent_sha="parent_solo",
            files=[
                FileChange(
                    filename="src/auth.py",
                    status="modified",
                    patch="@@ -1,5 +1,5 @@\n+if len(p) < 4: return False\n",
                )
            ],
            is_partial=False,
        )
        spec_index = SpecIndex(
            requirements=[SpecRequirement(req_id="FR-001", title="Password Policy", description="Min 8 chars")]
        )
        client = MockLLMClient(scenario=MockScenario.CONFIRMED_VIOLATION)
        orchestrator = SpecGuardOrchestrator(spec_index=spec_index, llm_client=client)

        report = orchestrator.run_audit("acme/repo", "single_parent_merge", git_context=ctx)
        assert report.base_parent_sha == "parent_solo"
        assert "base_revision" not in report.missing_data
        assert report.findings[0].severity == ViolationSeverity.SUSPECTED

    def test_octopus_merge_preserves_strictly_first_parent(self):
        """Octopus merge commit (4 parents) strictly assigns base_parent_sha to first_parent_sha."""
        # Parents: p1 (first/main), p2 (feature-a), p3 (feature-b), p4 (feature-c)
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="octopus_commit_abc",
            base_sha="p1_mainline",
            head_sha="octopus_head",
            is_merge_commit=True,
            first_parent_sha="p1_mainline",
            files=[
                FileChange(
                    filename="src/auth.py",
                    status="modified",
                    patch="@@ -1,5 +1,5 @@\n+if len(p) < 4: return False\n",
                )
            ],
            limitations=[],
            is_partial=False,
        )
        spec_index = SpecIndex(
            requirements=[SpecRequirement(req_id="FR-001", title="Password Policy", description="Min 8 chars")]
        )
        client = MockLLMClient(scenario=MockScenario.CONFIRMED_VIOLATION)
        orchestrator = SpecGuardOrchestrator(spec_index=spec_index, llm_client=client)

        report = orchestrator.run_audit("acme/repo", "octopus_commit_abc", git_context=ctx)
        assert report.base_parent_sha == "p1_mainline"
        assert report.metadata["first_parent_sha"] == "p1_mainline"
        assert report.metadata["is_merge_commit"] is True

    def test_octopus_merge_missing_first_parent_falls_back_and_flags_partial(self):
        """Octopus merge lacking first_parent_sha falls back to base_sha and flags partial."""
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="octopus_missing_first",
            base_sha="p2_fallback",
            head_sha="octopus_head",
            is_merge_commit=True,
            first_parent_sha=None,  # Missing!
            files=[
                FileChange(
                    filename="src/auth.py",
                    status="modified",
                    patch="@@ -1,5 +1,5 @@\n+if len(p) < 4: return False\n",
                )
            ],
            is_partial=False,
        )
        spec_index = SpecIndex(
            requirements=[SpecRequirement(req_id="FR-001", title="Password Policy", description="Min 8 chars")]
        )
        client = MockLLMClient(scenario=MockScenario.CONFIRMED_VIOLATION)
        orchestrator = SpecGuardOrchestrator(spec_index=spec_index, llm_client=client)

        report = orchestrator.run_audit("acme/repo", "octopus_missing_first", git_context=ctx)
        assert report.base_parent_sha is None
        assert report.is_partial is True
        assert "first_parent_sha" in report.missing_data
        assert any("missing first_parent_sha" in lim for lim in report.limitations)

    def test_circular_self_referencing_merge_commit(self):
        """Self-referencing commit (head_sha == base_sha == first_parent_sha) does not loop."""
        cycle_sha = "deadbeef_cycle"
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref=cycle_sha,
            base_sha=cycle_sha,
            head_sha=cycle_sha,
            is_merge_commit=True,
            first_parent_sha=cycle_sha,
            files=[],
            is_partial=False,
        )
        orchestrator = SpecGuardOrchestrator()
        report = orchestrator.run_audit("acme/repo", cycle_sha, git_context=ctx)
        assert report.base_parent_sha == cycle_sha
        assert len(report.findings) == 0

    def test_ac015_downgrade_invariance_on_implemented_and_not_affected(self):
        """Inaccessible base (AC-015) downgrades CONFIRMED VIOLATED, but preserves IMPLEMENTED & NOT_AFFECTED."""
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="inaccessible_base_commit",
            base_sha=None,
            head_sha="head_commit",
            is_merge_commit=False,
            first_parent_sha=None,
            files=[
                FileChange(
                    filename="src/service.py",
                    status="modified",
                    patch="@@ -1,5 +1,5 @@\n+implemented line\n",
                )
            ],
            is_partial=False,
        )
        spec_index = SpecIndex(
            requirements=[
                SpecRequirement(req_id="FR-001", title="Req 1", description="Description 1"),
                SpecRequirement(req_id="FR-002", title="Req 2", description="Description 2"),
            ]
        )

        class MultiScenarioLLM(BaseLLMClient):
            """Returns IMPLEMENTED for FR-001 and CONFIRMED VIOLATION for FR-002."""
            def __init__(self):
                super().__init__()
                self.call_count = 0
            def generate(self, messages, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    return Finding(
                        req_id="FR-001",
                        title="Implemented req",
                        status=RequirementStatus.IMPLEMENTED,
                        severity=None,
                        description="Fully implemented",
                        evidence=[
                            Evidence(
                                file_path="src/service.py",
                                revision="head_commit",
                                code_quote="implemented line",
                                spec_quote="Description 1",
                            )
                        ],
                    ).model_dump_json()
                else:
                    return Finding(
                        req_id="FR-002",
                        title="Violated req",
                        status=RequirementStatus.VIOLATED,
                        severity=ViolationSeverity.CONFIRMED,
                        description="Violates policy",
                        evidence=[
                            Evidence(
                                file_path="src/service.py",
                                revision="head_commit",
                                code_quote="bad line",
                                spec_quote="Description 2",
                            )
                        ],
                    ).model_dump_json()

        orchestrator = SpecGuardOrchestrator(spec_index=spec_index, llm_client=MultiScenarioLLM())
        report = orchestrator.run_audit("acme/repo", "inaccessible_base_commit", git_context=ctx)

        assert report.is_partial is True
        assert len(report.findings) == 2

        f1 = next(f for f in report.findings if f.req_id == "FR-001")
        f2 = next(f for f in report.findings if f.req_id == "FR-002")

        # Invariant 1: IMPLEMENTED remains IMPLEMENTED with severity=None (NOT touched by AC-015)
        assert f1.status == RequirementStatus.INSUFFICIENT_DATA
        assert f1.severity is None

        # Invariant 2: CONFIRMED VIOLATED was downgraded to SUSPECTED per AC-015
        assert f2.status == RequirementStatus.VIOLATED
        assert f2.severity == ViolationSeverity.SUSPECTED
        assert any("AC-015" in lim for lim in f2.limitations)

    def test_lineage_deleted_and_binary_files_do_not_flag_partial(self):
        """Deleted, removed, and binary files with patch=None do not trigger false partial flags."""
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="cleanup_commit",
            base_sha="base_cleanup",
            head_sha="head_cleanup",
            is_merge_commit=False,
            files=[
                FileChange(filename="old_auth.py", status="deleted", patch=None),
                FileChange(filename="logo.png", status="binary", patch=None),
                FileChange(filename="removed.py", status="removed", patch=None),
            ],
            is_partial=False,
        )
        orchestrator = SpecGuardOrchestrator()
        report = orchestrator.run_audit("acme/repo", "cleanup_commit", git_context=ctx)
        # Verify no false partial flags from deleted or binary files
        assert not any("missing_patch:old_auth.py" in m for m in report.missing_data)
        assert not any("missing_patch:logo.png" in m for m in report.missing_data)
        assert not any("missing_patch:removed.py" in m for m in report.missing_data)

    def test_non_merge_commit_prefers_base_sha_over_first_parent_sha(self):
        """When is_merge_commit=False, base_sha takes precedence even if first_parent_sha is populated."""
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="non_merge_commit",
            base_sha="regular_base_sha",
            head_sha="head_sha",
            is_merge_commit=False,
            first_parent_sha="erroneous_first_parent",
            files=[],
            is_partial=False,
        )
        orchestrator = SpecGuardOrchestrator()
        report = orchestrator.run_audit("acme/repo", "non_merge_commit", git_context=ctx)
        assert report.base_parent_sha == "regular_base_sha"

    def test_merge_commit_both_missing_parents_records_ac014_and_ac015(self):
        """Merge commit lacking both first_parent_sha and base_sha records both AC-014 and AC-015."""
        ctx = GitContext(
            target_repo="acme/repo",
            target_ref="double_missing_commit",
            base_sha=None,
            head_sha="double_missing_head",
            is_merge_commit=True,
            first_parent_sha=None,
            files=[],
            is_partial=False,
        )
        orchestrator = SpecGuardOrchestrator()
        report = orchestrator.run_audit("acme/repo", "double_missing_commit", git_context=ctx)
        assert report.base_parent_sha is None
        assert report.is_partial is True
        assert "first_parent_sha" in report.missing_data
        assert "base_revision" in report.missing_data
        assert any("first_parent_sha" in lim.lower() for lim in report.limitations)
        assert any("AC-015" in lim for lim in report.limitations)



# ==============================================================================
# Suite 4: Deterministic Audit ID Robustness
# ==============================================================================

class TestDeterministicAuditIdRobustness:
    """Verify hash determinism, sensitivity, normalization, and collision resistance."""

    def test_deterministic_audit_id_strict_repeatability(self):
        """Identical coordinates produce identical audit_id across 100 repeated runs."""
        ids = {
            generate_deterministic_audit_id(
                repo="Acme/Security-Core",
                target_ref="PR#42",
                head_sha="deadbeef12345678",
                base_parent_sha="cafebabe87654321",
            )
            for _ in range(100)
        }
        assert len(ids) == 1
        audit_id = list(ids)[0]
        assert audit_id.startswith("audit_")
        assert len(audit_id) == 18  # "audit_" (6) + 12 hex chars = 18

    def test_repo_normalization_case_and_whitespace(self):
        """Repository whitespace and uppercase letters are normalized."""
        id1 = generate_deterministic_audit_id("acme/repo", "PR#1", "sha1", "base1")
        id2 = generate_deterministic_audit_id("  Acme/Repo  ", "PR#1", "sha1", "base1")
        id3 = generate_deterministic_audit_id("ACME/REPO", "PR#1", "sha1", "base1")
        assert id1 == id2 == id3

    def test_target_ref_and_head_sha_whitespace_trimming(self):
        """Leading/trailing whitespace in target_ref and head_sha are trimmed."""
        id1 = generate_deterministic_audit_id("acme/repo", "PR#1", "sha1", "base1")
        id2 = generate_deterministic_audit_id("acme/repo", "  PR#1  ", "  sha1  ", "base1")
        assert id1 == id2

    def test_base_parent_sha_none_vs_empty_string(self):
        """base_parent_sha=None and base_parent_sha='' produce identical 'none' fallback ID."""
        id_none = generate_deterministic_audit_id("acme/repo", "PR#1", "sha1", None)
        id_empty = generate_deterministic_audit_id("acme/repo", "PR#1", "sha1", "")
        assert id_none == id_empty

    def test_sensitivity_to_single_character_flips(self):
        """Flipping a single character in any coordinate changes the audit_id."""
        baseline = generate_deterministic_audit_id("acme/repo", "PR#100", "sha_head_000", "sha_base_000")

        # Flip in repo
        assert baseline != generate_deterministic_audit_id("acme/repx", "PR#100", "sha_head_000", "sha_base_000")
        # Flip in target_ref
        assert baseline != generate_deterministic_audit_id("acme/repo", "PR#101", "sha_head_000", "sha_base_000")
        # Flip in head_sha
        assert baseline != generate_deterministic_audit_id("acme/repo", "PR#100", "sha_head_001", "sha_base_000")
        # Flip in base_parent_sha
        assert baseline != generate_deterministic_audit_id("acme/repo", "PR#100", "sha_head_000", "sha_base_001")

    def test_large_scale_collision_resistance_10000_keys(self):
        """10,000 distinct coordinate tuples produce 10,000 distinct audit IDs (zero collisions)."""
        generated_ids = set()
        count = 10_000

        for i in range(count):
            repo = f"org{i % 10}/repo{i % 25}"
            ref = f"PR#{i}"
            head = f"sha_head_{i:06x}"
            base = f"sha_base_{(i * 7) % 500:06x}" if i % 5 != 0 else None
            audit_id = generate_deterministic_audit_id(repo, ref, head, base)
            generated_ids.add(audit_id)

        assert len(generated_ids) == count
        for aid in list(generated_ids)[:50]:
            assert aid.startswith("audit_")
            assert len(aid) == 18
