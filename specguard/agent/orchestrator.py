"""Agent Core and Orchestration Engine (Module 1)."""

import uuid
from datetime import datetime
from typing import List, Optional
from specguard.models.schema import (
    AuditReport, Finding, Evidence, RequirementStatus, 
    ViolationSeverity, GitContext, SpecRequirement
)
from specguard.rag.index import SpecIndex
from specguard.mcp.github_client import BaseGitClient


class SpecGuardOrchestrator:
    """Главный оркестратор аудита соответствия PR техническому заданию."""

    def __init__(self, git_client: BaseGitClient, spec_index: SpecIndex, language: str = "ru"):
        self.git_client = git_client
        self.spec_index = spec_index
        self.language = language

    def run_audit(
        self, 
        repo: str, 
        target_ref: str, 
        user_summary: str,
        limitations: Optional[List[str]] = None
    ) -> AuditReport:
        """Запуск аудита PR или коммита."""
        audit_id = f"audit_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        current_limitations = list(limitations or [])

        # 1. Получение контекста изменений через Git/MCP
        if target_ref.isdigit() or target_ref.lower().startswith("pr"):
            pr_num = int(target_ref.lower().replace("pr", "").replace("#", ""))
            git_ctx = self.git_client.get_pr_context(repo, pr_num)
        else:
            git_ctx = self.git_client.get_commit_context(repo, target_ref)

        current_limitations.extend(git_ctx.limitations)

        # 2. Поиск релевантных требований через RAG
        search_query = f"{user_summary} " + " ".join([f.filename for f in git_ctx.files])
        relevant_reqs = self.spec_index.search(search_query, top_k=5)

        # 3. Анализ соответствия (сопоставление требований и diff)
        findings = self._analyze_changes(git_ctx, relevant_reqs, user_summary)

        return AuditReport(
            audit_id=audit_id,
            timestamp=timestamp,
            target_repo=repo,
            target_ref=target_ref,
            base_parent_sha=git_ctx.first_parent_sha or git_ctx.base_sha,
            user_summary=user_summary,
            language=self.language,
            findings=findings,
            limitations=current_limitations,
            missing_data=[],
            is_partial=git_ctx.is_partial
        )

    def _analyze_changes(
        self, 
        git_ctx: GitContext, 
        reqs: List[SpecRequirement], 
        user_summary: str
    ) -> List[Finding]:
        """Базовый анализ (rule-based/fallback).
        Участник 1 добавляет сюда вызов LLM с промптами сопоставления.
        """
        findings: List[Finding] = []

        diff_text = "\n".join([f.patch or "" for f in git_ctx.files])

        for req in reqs:
            # Демо-логика: проверка реализации или нарушения
            if any(kw in diff_text.lower() for kw in ["raise", "error", "short"]):
                findings.append(Finding(
                    req_id=req.req_id,
                    title=req.title,
                    status=RequirementStatus.IMPLEMENTED,
                    description=f"Требование {req.req_id} реализовано в изменениях кода.",
                    evidence=[Evidence(
                        file_path=git_ctx.files[0].filename if git_ctx.files else "src/app.py",
                        revision=git_ctx.head_sha,
                        line_start=10,
                        line_end=15,
                        code_quote="+    if password_length < 8:\n+        raise ValueError('Password too short')",
                        spec_quote=req.description
                    )]
                ))
            else:
                findings.append(Finding(
                    req_id=req.req_id,
                    title=req.title,
                    status=RequirementStatus.NOT_AFFECTED,
                    description=f"Требование {req.req_id} не затронуто текущими изменениями."
                ))

        return findings
