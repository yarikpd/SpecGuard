"""Build explicit saved audit materials for dispute test fixtures."""
from specguard.models.schema import AuditReport, FileChange, GitContext, SpecRequirement


def make_saved_report(**kwargs):
    findings = kwargs.get("findings", [])
    requirements = []
    files = []
    head = next((e.revision for f in findings for e in f.evidence if e.revision), "test-head")
    for finding in findings:
        quote = next((e.spec_quote for e in finding.evidence if e.spec_quote), finding.description)
        requirements.append(SpecRequirement(req_id=finding.req_id, title=finding.title, description=quote, raw_text=quote))
        for ev in finding.evidence + finding.counter_evidence:
            if not ev.code_quote:
                continue
            lines = ev.code_quote.splitlines()
            start = ev.line_start or 1
            patch = f"@@ -{start},0 +{start},{len(lines)} @@\n" + "\n".join(
                line if line.startswith('+') else '+' + line for line in lines
            )
            files.append(FileChange(filename=ev.file_path, status="modified", patch=patch))
    kwargs.setdefault("saved_requirements", requirements)
    kwargs.setdefault("saved_git_context", GitContext(target_repo=kwargs.get("target_repo", "test/repo"),
        target_ref=kwargs.get("target_ref", "test"), base_sha=kwargs.get("base_parent_sha"), head_sha=head, files=files))
    return AuditReport(**kwargs)
