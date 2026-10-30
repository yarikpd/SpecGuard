"""Shared fixtures and test helpers for SpecGuard 4-Tier Opaque-Box E2E Test Suite.

Provides realistic requirements, Git contexts, mock LLM/Git clients, and orchestrator
fixtures satisfying 100% offline test execution (R5).
"""

from __future__ import annotations

from tests.report_fixtures import make_saved_report

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import pytest

from specguard.agent.llm_client import MockLLMClient, MockScenario
from specguard.agent.orchestrator import SpecGuardOrchestrator
from specguard.mcp.github_client import MockGitClient
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


@pytest.fixture
def sample_requirements() -> List[SpecRequirement]:
    """Catalog of realistic specification requirements for E2E testing."""
    return [
        SpecRequirement(
            req_id="FR-001",
            title="Валидация длины пароля",
            section="Безопасность учетных записей",
            description="Длина пароля пользователя должна составлять не менее 8 символов.",
            conditions=["При регистрации пользователя", "При смене пароля"],
            exceptions=["Одноразовые токены авторизации (OTP)"],
            related_req_ids=["NFR-002"],
            raw_text="FR-001: Длина пароля должна быть не менее 8 символов.",
        ),
        SpecRequirement(
            req_id="FR-002",
            title="Поддержка форматов технического задания",
            section="Загрузка спецификаций",
            description="Система обязана поддерживать разбор файлов TXT, Markdown, DOCX и PDF.",
            conditions=["Файл содержит валидную разметку"],
            exceptions=["Сканированные изображения без OCR слоя"],
            related_req_ids=["FR-003"],
            raw_text="FR-002: Система поддерживает форматы TXT, Markdown, DOCX и текстовый PDF.",
        ),
        SpecRequirement(
            req_id="FR-003",
            title="Автоматический аудит PR и коммитов",
            section="Оркестрация анализа",
            description="Оркестратор должен принимать контекст изменений и формировать структурированный отчёт.",
            conditions=["Указан репозиторий и ссылка на PR/коммит"],
            exceptions=[],
            related_req_ids=["FR-004", "AC-014"],
            raw_text="FR-003: Оркестратор выполняет аудит изменений кода против требований ТЗ.",
        ),
        SpecRequirement(
            req_id="FR-004",
            title="Формирование отчета аудита с доказательной базой",
            section="Отчетность",
            description="Отчет должен содержать вердикт, цитаты из кода и ТЗ, номера строк и ревизии.",
            conditions=["Анализ успешно завершен"],
            exceptions=[],
            related_req_ids=["FR-008"],
            raw_text="FR-004: Отчет AuditReport содержит список Finding с точными Evidence.",
        ),
        SpecRequirement(
            req_id="FR-008",
            title="Градация серьезности нарушений",
            section="Классификация дефектов",
            description="Нарушения должны разделяться на подтвержденные (CONFIRMED) и спорные (SUSPECTED).",
            conditions=["Статус равен VIOLATED"],
            exceptions=["Статусы IMPLEMENTED и NOT_AFFECTED не имеют серьезности"],
            related_req_ids=["FR-004", "AC-015"],
            raw_text="FR-008: Нарушение классифицируется как CONFIRMED либо SUSPECTED.",
        ),
        SpecRequirement(
            req_id="FR-009",
            title="Обработка больших изменений и усечения diff",
            section="Масштабируемость",
            description="Большие diff не отклоняются, а анализируются частично с фиксацией ограничений.",
            conditions=["Размер diff превышает лимит символов"],
            exceptions=[],
            related_req_ids=["FR-013"],
            raw_text="FR-009: Большие изменения анализируются частично с флагом is_partial=True.",
        ),
        SpecRequirement(
            req_id="FR-012",
            title="Процедура оспаривания замечаний",
            section="Арбитраж",
            description="Пользователь может оспорить замечание через DisputeRequest на сохраненных данных.",
            conditions=["Указан корректный audit_id и finding_req_id"],
            exceptions=[],
            related_req_ids=["FR-004"],
            raw_text="FR-012: Повторный пересмотр замечания выполняется по комментарию пользователя.",
        ),
        SpecRequirement(
            req_id="FR-013",
            title="Фиксация неполных данных и ограничений",
            section="Надежность",
            description="При неполных данных отчет помечается флагом is_partial=True со списком missing_data.",
            conditions=["Недоступны файлы или базовая ревизия"],
            exceptions=[],
            related_req_ids=["FR-009", "AC-015"],
            raw_text="FR-013: Неполные данные не представляются как проверенные.",
        ),
        SpecRequirement(
            req_id="NFR-002",
            title="Сетевые таймауты и защита от Prompt Injection",
            section="Информационная безопасность",
            description="Таймаут сетевых соединений должен быть не менее 15с. Недоверенные данные изолируются в raw-тегах.",
            conditions=["Обработка входящих diff и комментариев"],
            exceptions=[],
            related_req_ids=["FR-001"],
            raw_text="NFR-002: Таймаут не менее 15с. Защита от prompt injection через изоляцию сырых данных.",
        ),
        SpecRequirement(
            req_id="AC-014",
            title="Отслеживание первого родителя для merge-коммитов",
            section="Git интеграция",
            description="Для merge-коммитов базовой ревизией фиксируется SHA первого родителя (first_parent_sha).",
            conditions=["Коммит имеет флаг is_merge_commit=True"],
            exceptions=[],
            related_req_ids=["FR-003"],
            raw_text="AC-014: Для merge-коммита base_parent_sha фиксирует first_parent_sha.",
        ),
        SpecRequirement(
            req_id="AC-015",
            title="Обработка недоступной базовой ревизии",
            section="Устойчивость к сбоям",
            description="При отсутствии базовой ревизии подтвержденные нарушения понижаются до SUSPECTED.",
            conditions=["base_parent_sha is None"],
            exceptions=[],
            related_req_ids=["FR-008", "FR-013"],
            raw_text="AC-015: Недоступная базовая ревизия понижает статус нарушений до SUSPECTED.",
        ),
    ]


@pytest.fixture
def sample_spec_index(sample_requirements: List[SpecRequirement]) -> SpecIndex:
    """Pre-indexed SpecIndex with sample requirements."""
    index = SpecIndex()
    index.build_index(sample_requirements)
    return index


@pytest.fixture
def empty_spec_index() -> SpecIndex:
    """Empty SpecIndex for testing degradation and boundary conditions."""
    index = SpecIndex()
    index.build_index([])
    return index


@pytest.fixture
def make_file_change() -> Callable[..., FileChange]:
    """Helper factory for FileChange instances."""
    def _create(
        filename: str = "src/auth.py",
        patch: Optional[str] = "@@ -1,3 +1,5 @@\n+def validate():\n+    pass",
        status: str = "modified",
        additions: int = 2,
        deletions: int = 0,
    ) -> FileChange:
        return FileChange(
            filename=filename,
            patch=patch,
            status=status,
            additions=additions,
            deletions=deletions,
        )
    return _create


@pytest.fixture
def make_evidence() -> Callable[..., Evidence]:
    """Helper factory for Evidence instances."""
    def _create(
        file_path: str = "src/auth.py",
        revision: str = "head_def456",
        line_start: Optional[int] = 10,
        line_end: Optional[int] = 12,
        code_quote: str = "+    if len(password) < 8: raise ValueError()",
        spec_quote: str = "FR-001: Длина пароля должна быть не менее 8 символов.",
    ) -> Evidence:
        return Evidence(
            file_path=file_path,
            revision=revision,
            line_start=line_start,
            line_end=line_end,
            code_quote=code_quote,
            spec_quote=spec_quote,
        )
    return _create


@pytest.fixture
def make_finding(make_evidence: Callable[..., Evidence]) -> Callable[..., Finding]:
    """Helper factory for Finding instances."""
    def _create(
        req_id: str = "FR-001",
        title: str = "Валидация пароля",
        status: RequirementStatus = RequirementStatus.IMPLEMENTED,
        severity: Optional[ViolationSeverity] = None,
        description: str = "Требование корректно реализовано.",
        evidence: Optional[List[Evidence]] = None,
        counter_evidence: Optional[List[Evidence]] = None,
        limitations: Optional[List[str]] = None,
    ) -> Finding:
        ev_list = evidence if evidence is not None else [make_evidence()]
        return Finding(
            req_id=req_id,
            title=title,
            status=status,
            severity=severity,
            description=description,
            evidence=ev_list,
            counter_evidence=counter_evidence or [],
            limitations=limitations or [],
        )
    return _create


@pytest.fixture
def make_git_context(make_file_change: Callable[..., FileChange]) -> Callable[..., GitContext]:
    """Helper factory for GitContext instances."""
    def _create(
        target_repo: str = "octocat/SpecGuardRepo",
        target_ref: str = "PR#42",
        head_sha: str = "head_def456",
        base_sha: Optional[str] = "base_abc123",
        is_merge_commit: bool = False,
        first_parent_sha: Optional[str] = None,
        files: Optional[List[FileChange]] = None,
        limitations: Optional[List[str]] = None,
        is_partial: bool = False,
    ) -> GitContext:
        file_list = files if files is not None else [make_file_change()]
        return GitContext(
            target_repo=target_repo,
            target_ref=target_ref,
            head_sha=head_sha,
            base_sha=base_sha,
            is_merge_commit=is_merge_commit,
            first_parent_sha=first_parent_sha,
            files=file_list,
            limitations=limitations or [],
            is_partial=is_partial,
        )
    return _create


@pytest.fixture
def make_audit_report(make_finding: Callable[..., Finding]) -> Callable[..., AuditReport]:
    """Helper factory for AuditReport instances."""
    def _create(
        audit_id: str = "audit_fixture_001",
        target_repo: str = "octocat/SpecGuardRepo",
        target_ref: str = "PR#42",
        base_parent_sha: Optional[str] = "base_abc123",
        findings: Optional[List[Finding]] = None,
        limitations: Optional[List[str]] = None,
        missing_data: Optional[List[str]] = None,
        is_partial: bool = False,
        language: str = "ru",
        user_summary: str = "Реализация валидации пароля по FR-001",
    ) -> AuditReport:
        f_list = findings if findings is not None else [make_finding()]
        return make_saved_report(
            audit_id=audit_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            target_repo=target_repo,
            target_ref=target_ref,
            base_parent_sha=base_parent_sha,
            user_summary=user_summary,
            language=language,
            findings=f_list,
            limitations=limitations or [],
            missing_data=missing_data or [],
            is_partial=is_partial,
            metadata={"test_fixture": True},
        )
    return _create


@pytest.fixture
def make_dispute_request() -> Callable[..., DisputeRequest]:
    """Helper factory for DisputeRequest instances."""
    def _create(
        audit_id: str = "audit_fixture_001",
        finding_req_id: str = "FR-001",
        user_comment: str = "Проверка пароля вынесена в src/auth/validators.py, замечание некорректно.",
    ) -> DisputeRequest:
        return DisputeRequest(
            audit_id=audit_id,
            finding_req_id=finding_req_id,
            user_comment=user_comment,
        )
    return _create


@pytest.fixture
def sample_git_client() -> MockGitClient:
    """Standard MockGitClient with realistic files."""
    return MockGitClient(
        sample_files=[
            FileChange(
                filename="src/auth.py",
                status="modified",
                patch="@@ -10,4 +10,6 @@\n+    if password_length < 8:\n+        raise ValueError('Password too short')",
                additions=2,
                deletions=0,
            ),
            FileChange(
                filename="src/config.py",
                status="modified",
                patch="@@ -1,2 +1,3 @@\n+TIMEOUT_SECONDS = 30",
                additions=1,
                deletions=0,
            ),
        ]
    )


@pytest.fixture
def sample_llm_client() -> MockLLMClient:
    """Standard deterministic MockLLMClient in AUTO mode."""
    return MockLLMClient(scenario=MockScenario.AUTO)


@pytest.fixture
def sample_orchestrator(
    sample_git_client: MockGitClient,
    sample_spec_index: SpecIndex,
    sample_llm_client: MockLLMClient,
) -> SpecGuardOrchestrator:
    """Configured SpecGuardOrchestrator instance ready for testing."""
    return SpecGuardOrchestrator(
        git_client=sample_git_client,
        spec_index=sample_spec_index,
        llm_client=sample_llm_client,
        language="ru",
    )
