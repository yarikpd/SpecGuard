"""Application operations shared by command mode and the terminal UI."""
from contextlib import ExitStack
import math
from pathlib import Path
from uuid import uuid4

from specguard.agent.llm_client import MockLLMClient, OpenAILLMClient, extract_json
from specguard.agent.orchestrator import SpecGuardOrchestrator
from specguard.agent.scope import review_change_scope
from specguard.cli.config import AppConfig, load_github_token, load_mcp_server
from specguard.cli.history import HistoryStore
from specguard.mcp.github_client import GitHubMCPClient, MockGitHubClient, parse_repo
from specguard.models.schema import DisputeRequest
from specguard.rag.embeddings import MockEmbeddingClient, OpenAIEmbeddingClient
from specguard.rag.service import RAGService
from specguard.rag.index import HybridSpecIndex
from collections import Counter


def _close_with(stack, client):
    if hasattr(client, "close"):
        stack.callback(client.close)
    return client


def llm_client(config, demo=False):
    if demo:
        return MockLLMClient()
    config.llm.validate()
    return OpenAILLMClient(base_url=config.llm.base_url, model=config.llm.model,
                           api_key=config.llm.api_key or None)


def check_model_connection(kind, config):
    """Exercise the same endpoints and response formats used by audits."""
    if kind not in ("llm", "embedding"):
        raise ValueError("Неизвестный тип модели")
    config.validate()
    with ExitStack() as stack:
        if kind == "llm":
            client = _close_with(stack, OpenAILLMClient(
                base_url=config.base_url, model=config.model, api_key=config.api_key or None,
                timeout=30, max_retries=0))
            response = client.generate([{"role": "user", "content": 'Ответь JSON-объектом: {"ok": true}'}],
                                       response_format="json")
            if not isinstance(extract_json(response), dict):
                raise ValueError("API не вернул JSON-объект")
            return "LLM API вернул JSON-объект. Качество модели не проверялось."
        client = _close_with(stack, OpenAIEmbeddingClient(
            base_url=config.base_url, model_name=config.model, api_key=config.api_key or None,
            timeout=30))
        vector = client.embed_text("SpecGuard connection test")
        if not vector or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in vector):
            raise ValueError("API вернул некорректный вектор")
        return f"Embedding API вернул вектор размерности {len(vector)}. Качество модели не проверялось."


def run_audit(config: AppConfig, store: HistoryStore, spec: str, repo: str,
              target: str, summary: str, lang="ru", demo=False, progress=lambda text: None):
    if lang not in ("ru", "en"):
        raise ValueError("Язык отчёта: ru или en")
    owner, name = parse_repo(repo)
    if not target.strip():
        raise ValueError("Выберите PR или укажите SHA коммита")
    specs = [spec] if isinstance(spec, (str, Path)) else list(spec)
    if not specs or any(not Path(item).is_file() for item in specs if not isinstance(item, dict)):
        raise ValueError("Файл ТЗ не найден")
    if not demo:
        config.llm.validate()
        config.embedding.validate()
    with ExitStack() as stack:
        llm = _close_with(stack, llm_client(config, demo))
        embedding = _close_with(stack, MockEmbeddingClient() if demo else OpenAIEmbeddingClient(
            base_url=config.embedding.base_url, model_name=config.embedding.model,
            api_key=config.embedding.api_key or None))
        git = _close_with(stack, MockGitHubClient() if demo else GitHubMCPClient(mcp_server_cmd=load_mcp_server() or None, auth_token=load_github_token()))
        progress("Извлечение и индексация ТЗ…")
        rag = RAGService(embedding, storage_dir=store.directory / "snapshots")
        snapshots, warnings, requirements, spec_paths = [], [], [], []
        for item in specs:
            if isinstance(item, dict):
                snapshot_id = item.get("snapshot_id", "")
                if not isinstance(snapshot_id, str) or len(snapshot_id) != 64 or any(c not in "0123456789abcdef" for c in snapshot_id):
                    raise ValueError("Некорректный идентификатор снимка ТЗ")
                snapshot = rag.snapshot_store.load_snapshot(snapshot_id)
                if snapshot is None:
                    raise ValueError("Снимок ТЗ из истории не найден. Выберите исходный файл заново.")
                path = item["path"]
                document_warnings = snapshot.warnings
                document_index = HybridSpecIndex(embedding)
                document_index.build_index(snapshot.requirements)
                document_index.snapshot_id = snapshot.snapshot_id
                document_index.source_warnings = list(snapshot.warnings)
            else:
                path = str(Path(item).resolve())
                document_index, snapshot, document_warnings = rag.index_document(item)
            spec_paths.append(path)
            snapshots.append(snapshot)
            warnings.extend(f"{Path(path).name}: {warning}" for warning in document_warnings)
            requirements.extend(req.model_copy(deep=True, update={"source_file": path})
                                for req in snapshot.requirements)
        if len(snapshots) == 1:
            index = document_index
        else:
            # IDs are keys throughout retrieval and dispute; distinguish duplicates
            # rather than silently dropping a requirement from another document.
            counts = Counter(req.req_id.upper() for req in requirements)
            if any(count > 1 for count in counts.values()):
                warnings.append("Совпадающие идентификаторы требований различаются префиксами DOC1, DOC2 и т. д. по порядку выбранных файлов.")
            for req in requirements:
                if counts[req.req_id.upper()] > 1:
                    source = next(i for i, item in enumerate(spec_paths, 1)
                                  if str(Path(item).resolve()) == req.source_file)
                    req.req_id = f"DOC{source}-{req.req_id}"
                req.related_req_ids = [f"DOC{next(i for i, item in enumerate(spec_paths, 1) if str(Path(item).resolve()) == req.source_file)}-{rid}"
                                       if counts[rid.upper()] > 1 else rid for rid in req.related_req_ids]
            ids = [req.req_id.upper() for req in requirements]
            if len(ids) != len(set(ids)):
                raise ValueError("В одном документе повторяются идентификаторы требований; исправьте ТЗ.")
            index = HybridSpecIndex(embedding)
            index.build_index(requirements)
            index.snapshot_id = None
            index.source_warnings = warnings
        progress("Чтение изменений через MCP и анализ требований…")
        report = SpecGuardOrchestrator(git, index, llm, language=lang).run_audit(
            repo=f"{owner}/{name}", target_ref=target, user_summary=summary,
            language=lang, limitations=warnings, audit_id="audit_" + uuid4().hex)
        if len(snapshots) > 1:
            report.saved_requirements = [req.model_copy(deep=True) for req in requirements]
        report.spec_snapshot_id = snapshots[0].snapshot_id if len(snapshots) == 1 else None
        report.metadata["spec_snapshots"] = [{"path": str(Path(path).resolve()), "snapshot_id": snap.snapshot_id} for path, snap in zip(spec_paths, snapshots)]
        report.metadata.update({"mode": "demo" if demo else "live",
                                "llm_profile": {"model": "mock"} if demo else config.llm.public(),
                                "embedding_profile": {"model": "mock"} if demo else config.embedding.public()})
        if not demo:
            progress("Поиск изменений без найденной связи с выбранным ТЗ…")
            review_change_scope(report, index.requirements, llm)
        else:
            report.metadata["scope_review"] = "not_run_demo"
        store.save(report)
        progress("Отчёт сохранён")
        return report


def review(config, store, audit_id, req_id, comment, demo=False):
    if not comment.strip():
        raise ValueError("Введите комментарий к замечанию")
    report = store.get(audit_id)
    if report is None:
        raise ValueError("Аудит не найден")
    finding = next((f for f in report.findings if f.req_id.casefold() == req_id.casefold()), None)
    if finding is None:
        raise ValueError("Требование не найдено в отчёте")
    if not report.saved_requirements or report.saved_git_context is None:
        raise ValueError("Нет сохранённых исходных материалов для пересмотра")
    request = DisputeRequest(audit_id=audit_id, finding_req_id=finding.req_id, user_comment=comment)
    with ExitStack() as stack:
        llm = _close_with(stack, llm_client(config, demo))
        response = SpecGuardOrchestrator(llm_client=llm, language=report.language).review_dispute(
            request, report, language=report.language)
        path = store.save_dispute(request, response)
        return response, path
