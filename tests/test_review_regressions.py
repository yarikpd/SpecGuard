from pathlib import Path

import pytest

from specguard.agent.orchestrator import SpecGuardOrchestrator, _validate_finding
from specguard.agent.llm_client import MockLLMClient
from specguard.models.schema import (AuditReport, DisputeRequest, Evidence, FileChange,
    Finding, GitContext, RequirementStatus, SpecRequirement, ViolationSeverity)
from specguard.rag import MockEmbeddingClient, RAGService
from specguard.rag.extractor import RequirementExtractor
from specguard.mcp.github_client import GitHubMCPClient
from specguard.mcp.errors import GitSourceError, GitSourceErrorKind


def fixture_data():
    req = SpecRequirement(req_id="FR-001", description="Passwords require 8 characters", raw_text="Passwords require 8 characters")
    ctx = GitContext(target_repo="a/b", target_ref="abc", base_sha="base", head_sha="head",
        files=[FileChange(filename="auth.py", status="modified", patch="@@ -1,1 +1,1 @@\n-old\n+check_length(8)")])
    ev = Evidence(file_path="auth.py", revision="head", line_start=1, line_end=1,
        code_quote="check_length(8)", spec_quote="Passwords require 8 characters")
    return req, ctx, ev


def test_verifies_quote_path_revision_and_lines():
    req, ctx, ev = fixture_data()
    finding = Finding(req_id=req.req_id, title="Password", description="ok", status=RequirementStatus.IMPLEMENTED, evidence=[ev])
    assert _validate_finding(finding, req, ctx).status == RequirementStatus.IMPLEMENTED
    for changes in ({"revision": "wrong"}, {"line_start": 2, "line_end": 2}, {"file_path": "other.py"}, {"code_quote": "invented"}, {"spec_quote": "invented"}):
        bad = finding.model_copy(update={"evidence": [ev.model_copy(update=changes)]})
        checked = _validate_finding(bad, req, ctx)
        assert checked.status == RequirementStatus.INSUFFICIENT_DATA
        assert checked.evidence == []


def test_dispute_uses_saved_full_context_and_requirement():
    req, ctx, ev = fixture_data()
    finding = Finding(req_id=req.req_id, title="Password", description="bad", status=RequirementStatus.VIOLATED, severity=ViolationSeverity.SUSPECTED, evidence=[ev])
    report = AuditReport(audit_id="a", target_repo="a/b", target_ref="abc", findings=[finding], saved_requirements=[req], saved_git_context=ctx)
    llm = MockLLMClient()
    orchestrator = SpecGuardOrchestrator(llm_client=llm)
    orchestrator.review_dispute(DisputeRequest(audit_id="a", finding_req_id="FR-001", user_comment="review"), report)
    assert "Passwords require 8 characters" in llm.last_prompt
    assert "+check_length(8)" in llm.last_prompt
    with pytest.raises(ValueError, match="materials"):
        orchestrator.review_dispute(DisputeRequest(audit_id="a", finding_req_id="FR-001", user_comment="review"), report.model_copy(update={"saved_git_context": None}))


def test_plain_paragraphs_survive_alongside_list():
    requirements = RequirementExtractor().extract("Independent requirement.\n\n- FR-001: Listed requirement.\n\nAnother independent requirement.")
    assert any("Independent requirement" in r.description for r in requirements)
    assert any("Another independent requirement" in r.description for r in requirements)


def test_snapshot_sources_and_warnings_are_not_reused(tmp_path, monkeypatch):
    from specguard.rag import service as module
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("same")
    b.write_text("same")
    service = RAGService(MockEmbeddingClient(), tmp_path / "snapshots")
    monkeypatch.setattr(module, "parse_document", lambda p: ("same", ["incomplete"] if p.name == "b.txt" else []))
    _, first, _ = service.index_document(a)
    index, second, warnings = service.index_document(b)
    assert first.snapshot_id != second.snapshot_id
    assert second.source_filename == "b.txt" and warnings == ["incomplete"]
    assert index.snapshot_id == second.snapshot_id and index.source_warnings == warnings


def test_live_client_never_returns_mock_data(monkeypatch):
    monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(GitSourceError) as error:
        GitHubMCPClient().get_pr_context("a/b", 1)
    assert error.value.kind == GitSourceErrorKind.NOT_CONFIGURED


def test_verified_audit_and_dispute_survive_serialization():
    from specguard.rag.index import SpecIndex

    req, ctx, ev = fixture_data()
    valid = Finding(req_id=req.req_id, title="Password", description="Verified change",
                    status=RequirementStatus.IMPLEMENTED, evidence=[ev])
    audit_llm = MockLLMClient(default_response=valid.model_dump(mode="json"))
    # Explicit queue prevents scenario dispatch from replacing the supplied response.
    audit_llm.enqueue_response(valid)
    engine = SpecGuardOrchestrator(spec_index=SpecIndex([req]), llm_client=audit_llm)
    report = engine.run_audit("a/b", "abc", git_context=ctx)
    assert report.findings[0].status == RequirementStatus.IMPLEMENTED
    assert report.findings[0].evidence == [ev]
    assert not report.is_partial

    original = report.model_copy(deep=True)
    original.findings[0].status = RequirementStatus.VIOLATED
    original.findings[0].severity = ViolationSeverity.SUSPECTED
    original = AuditReport.model_validate_json(original.model_dump_json())
    engine.spec_index = SpecIndex([req.model_copy(update={"description": "NEW REQUIREMENT TEXT", "raw_text": "NEW REQUIREMENT TEXT"})])
    dispute_llm = MockLLMClient()
    dispute_llm.enqueue_response({"audit_id": original.audit_id, "finding_req_id": req.req_id,
        "original_status": "VIOLATED", "original_severity": "SUSPECTED", "dispute_status": "OVERTURNED",
        "updated_finding": valid.model_dump(mode="json"), "reasoning": "The saved diff implements the requirement."})
    engine.llm_client = dispute_llm
    result = engine.review_dispute(DisputeRequest(audit_id=original.audit_id, finding_req_id=req.req_id,
                                                  user_comment="The change includes the check."), original)
    assert result.dispute_status == "OVERTURNED"
    assert result.updated_finding.status == RequirementStatus.IMPLEMENTED
    assert "Passwords require 8 characters" in dispute_llm.last_prompt
    assert "NEW REQUIREMENT TEXT" not in dispute_llm.last_prompt


def test_dispute_cannot_overturn_without_comparative_base():
    req, ctx, ev = fixture_data()
    finding = Finding(req_id=req.req_id, title="Password", description="bad", status=RequirementStatus.VIOLATED,
                      severity=ViolationSeverity.SUSPECTED, evidence=[ev])
    report = AuditReport(audit_id="a", target_repo="a/b", target_ref="abc", findings=[finding],
                         saved_requirements=[req], saved_git_context=ctx)
    llm = MockLLMClient()
    llm.enqueue_response({"audit_id":"a", "finding_req_id":"FR-001", "original_status":"VIOLATED",
        "original_severity":"SUSPECTED", "dispute_status":"OVERTURNED", "updated_finding": finding.model_copy(
            update={"status": RequirementStatus.IMPLEMENTED, "severity": None}).model_dump(mode="json"), "reasoning":"ok"})
    response = SpecGuardOrchestrator(llm_client=llm).review_dispute(
        DisputeRequest(audit_id="a", finding_req_id="FR-001", user_comment="review"), report)
    assert response.dispute_status == "MODIFIED"
    assert response.updated_finding.status == RequirementStatus.INSUFFICIENT_DATA


def test_invalid_counter_evidence_retries_and_preserves_other_findings():
    import json
    req, ctx, _ = fixture_data()
    bad = json.dumps(dict(req_id=req.req_id, title='Check', status='NOT_AFFECTED',
                          description='Unrelated', counter_evidence=['No changes in this PR']))
    good = json.dumps(dict(req_id=req.req_id, title='Check', status='NOT_AFFECTED',
                           description='Unrelated', counter_evidence=[]))
    class Client:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = 0
        def generate(self, *args, **kwargs):
            self.calls += 1
            return next(self.responses)
    client = Client([bad, good])
    core = SpecGuardOrchestrator(llm_client=client)
    findings = core._analyze_changes(ctx, [req], '', '', 'base')
    assert client.calls == 2
    assert findings[0].status == RequirementStatus.NOT_AFFECTED
    assert findings[0].counter_evidence == []
    client = Client([bad, bad, good])
    core = SpecGuardOrchestrator(llm_client=client)
    second = req.model_copy(update={'req_id': 'FR-002'})
    findings = core._analyze_changes(ctx, [req, second], '', '', 'base')
    assert findings[0].status == RequirementStatus.INSUFFICIENT_DATA
    assert 'invalid_model_response' in findings[0].limitations
    assert findings[1].status == RequirementStatus.NOT_AFFECTED


def test_repeated_schema_failure_returns_partial_report_with_saved_materials():
    from specguard.rag.index import SpecIndex
    req, ctx, _ = fixture_data()
    class Client:
        def generate(self, *args, **kwargs):
            return '{"counter_evidence": ["No source quote"]}'
    report = SpecGuardOrchestrator(spec_index=SpecIndex([req]), llm_client=Client()).run_audit(
        repo='a/b', target_ref='PR#6', git_context=ctx)
    assert report.is_partial
    assert 'valid_model_response' in report.missing_data
    assert report.findings[0].status == RequirementStatus.INSUFFICIENT_DATA
    assert report.saved_requirements and report.saved_git_context
