import pytest

from specguard.cli.history import HistoryStore
from specguard.models.schema import (
    AuditReport, DisputeRequest, DisputeResponse, Finding, RequirementStatus,
)


def report():
    return AuditReport(audit_id="audit_same", target_repo="owner/repo", target_ref="42")


def test_history_has_no_constructor_side_effect(tmp_path):
    store = HistoryStore(tmp_path / "history")
    assert store.list() == []
    assert not store.directory.exists()


def test_repeated_audits_preserve_both_runs(tmp_path):
    store = HistoryStore(tmp_path)
    first, second = report(), report()
    second.user_summary = "Updated explanation"
    store.save(first)
    store.save(second)
    assert first.audit_id != second.audit_id
    assert len(store.list()) == 2
    assert store.get(first.audit_id).user_summary == ""
    assert store.get(second.audit_id).user_summary == "Updated explanation"


@pytest.mark.parametrize("audit_id", ["../outside", "/tmp/file", "a/b", "..", ""])
def test_history_rejects_path_traversal(tmp_path, audit_id):
    store = HistoryStore(tmp_path)
    with pytest.raises(ValueError):
        store.get(audit_id)
    with pytest.raises(ValueError):
        store.save(report().model_copy(update={"audit_id": audit_id}))


def test_repeated_disputes_preserve_comments_and_original(tmp_path):
    store = HistoryStore(tmp_path)
    original = report()
    original_path = store.save(original)
    original_bytes = original_path.read_bytes()
    finding = Finding(req_id="FR-001", title="Check", description="Details",
                      status=RequirementStatus.VIOLATED)
    response = DisputeResponse(
        audit_id=original.audit_id, finding_req_id=finding.req_id,
        original_status=finding.status, dispute_status="UPHELD",
        updated_finding=finding, reasoning="Still applies",
    )
    for comment in ["First comment", "Second comment"]:
        request = DisputeRequest(audit_id=original.audit_id,
                                 finding_req_id=finding.req_id, user_comment=comment)
        store.save_dispute(request, response)
    entries = store.list_disputes(original.audit_id)
    assert {entry['request']['user_comment'] for entry in entries} == {"First comment", "Second comment"}
    assert original_path.read_bytes() == original_bytes
