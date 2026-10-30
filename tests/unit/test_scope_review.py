import json
from specguard.agent.scope import review_change_scope
from specguard.cli.render import render_report
from specguard.models.schema import AuditReport


class Client:
    def __init__(self, observations):
        self.observations = observations
    def generate(self, messages, **kwargs):
        self.messages = messages
        return json.dumps({'observations': self.observations})


def test_scope_review_with_saved_case6_materials():
    # Keep test self-contained: an unrelated added middleware alongside a spec-related change.
    from specguard.models.schema import GitContext, FileChange, SpecRequirement
    report = AuditReport(audit_id='scope', target_repo='owner/repo', target_ref='5', saved_git_context=GitContext(
        target_repo='owner/repo', target_ref='5', head_sha='head', base_sha='base', files=[FileChange(
            filename='app/main.py', status='modified', patch='@@ -0,0 +1,3 @@\n+@app.middleware("http")\n+async def log_requests(request, call_next):\n+    return await call_next(request)')]))
    req = SpecRequirement(req_id='AGE', description='Age must be at least 18')
    observation = {'title': 'Добавлен middleware логирования', 'description': 'Связь с выбранными требованиями не найдена.',
                   'evidence': [{'file_path': 'app/main.py', 'revision': 'head', 'line_start': 1, 'line_end': 2,
                                 'code_quote': '@app.middleware("http")\nasync def log_requests(request, call_next):'}]}
    client = Client([observation])
    review_change_scope(report, [req], client)
    assert report.scope_observations and not report.findings
    assert 'Age must be at least 18' in client.messages[1]['content']
    assert report.metadata['scope_review'] == 'completed'
    restored = AuditReport.model_validate_json(report.model_dump_json())
    assert restored.scope_observations[0].title == observation['title']
    panel = next(p for p in render_report(restored).renderables if getattr(p, 'title', '') == 'Изменения без найденной связи с ТЗ')
    assert 'не является нарушением' in panel.renderable.plain
    # Fabricated citations must not be surfaced as verified observations.
    observation['evidence'][0]['code_quote'] = 'invented_code()'
    report.scope_observations = []
    review_change_scope(report, [req], Client([observation]))
    assert not report.scope_observations
    assert report.is_partial and 'scope_evidence' in report.missing_data
