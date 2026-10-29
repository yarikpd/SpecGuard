# SpecGuard 🛡️

Автономный ИИ-агент для автоматического аудита соответствия Pull Request и коммитов в GitHub требованиям технического задания через RAG и GitHub MCP.

## Быстрый старт

### Установка

```bash
pip install -r requirements.txt
```

### Использование (пример)

```python
from specguard.rag import RAGService, MockEmbeddingClient

# Инициализация RAG-сервиса
client = MockEmbeddingClient(dimension=384)
service = RAGService(embedding_client=client)

# Индексация технического задания
index, snapshot, warnings = service.index_document("specification/TECHNICAL_SPEC.md")

# Поиск требований
results = index.search("аутентификация пользователя", top_k=5)
for req in results:
    print(f"{req.req_id}: {req.description}")

# Восстановление исторического снимка для оспаривания
historical_index, historical_snapshot = service.restore_index(snapshot.snapshot_id)
```

## Архитектура

Проект разделен на 6 независимых модулей:

- **Модуль 1** (`specguard/agent/`) — ✅ Ядро агента и оркестрация (100% готов)
- **Модуль 2** (`specguard/rag/`) — ✅ RAG & граф требований (100% готов)
- **Модуль 3** (`specguard/mcp/`) — GitHub MCP адаптер
- **Модуль 4** (`specguard/cli/`) — CLI интерфейс
- **Модуль 5** (`specguard/eval/`) — Эталонный датасет и метрики
- **Модуль 6** (`presentation/`) — Презентация и демо

### Модуль 1 (Agent Core & Orchestration) — Реализовано

Модуль ядра агента и оркестрации полностью завершен, верифицирован независимым Victory Audit и покрыт тестами (100% statement coverage ядра):

**Возможности:**
- **Оркестратор аудита (`SpecGuardOrchestrator`)**:
  - Прием контекста изменений Git (`GitContext`), поиск релевантных требований в ТЗ через RAG (`SpecIndex`), сопоставление через LLM и генерация структурированного отчета `AuditReport`.
  - Корректная обработка merge-коммитов: вычисление и фиксация SHA первого родителя (`is_merge_commit=True -> first_parent_sha -> base_parent_sha`, AC-014).
  - Обработка неполных и усеченных данных: деградация вердиктов до `SUSPECTED` при недоступной базовой ревизии (AC-015), выставление `is_partial=True`, наполнение списков `limitations` и `missing_data`.
  - Формирование точной доказательной базы (`Evidence`): путь к файлу, ревизия, диапазон строк, цитаты кода и спецификации.
- **Процедура оспаривания (`review_dispute`)**:
  - Пересмотр замечаний (`DisputeRequest` -> `DisputeResponse`) на сохраненных исходных материалах аудита без сетевых вызовов (FR-012, AC-008).
  - Защита от prompt injection в комментариях апелляции: вредоносные инструкции блокируются, замечание сохраняется (`UPHELD`).
- **Защита от Prompt Injection (`specguard/agent/defense.py`)**:
  - Изоляция diff, коммит-сообщений и фрагментов ТЗ внутри строгих XML-тегов с экранированием разделителей и тегов выхода (NFR-002, AC-009).
  - Детектирование русскоязычных и мультиязычных атак без ложных срабатываний на легитимный код.
- **Интеграция с LLM (`specguard/agent/llm_client.py`)**:
  - `OpenAILLMClient`: промышленный клиент с поддержкой OpenAI API и локального Ollama, ретраями, экспоненциальным backoff и парсингом JSON.
  - `MockLLMClient`: детерминированный оффлайн-движок с набором готовых сценариев для автономных тестов без API-ключей.

**Архитектура:**
```
SpecGuardOrchestrator (оркестратор)
  ├── run_audit() → GitContext + SpecIndex + BaseLLMClient → AuditReport
  ├── review_dispute() → DisputeRequest + AuditReport → DisputeResponse
  ├── Prompts & Defense (specguard/agent/defense.py, prompts.py)
  └── LLM Engine (specguard/agent/llm_client.py: OpenAI / Ollama / Mock)
```

**Пример использования оркестратора:**
```python
from specguard.agent.orchestrator import SpecGuardOrchestrator
from specguard.agent.llm_client import MockLLMClient
from specguard.mcp.github_client import MockGitClient
from specguard.rag.index import SpecIndex

# Инициализация с моками для оффлайн-работы (или боевыми клиентами)
orchestrator = SpecGuardOrchestrator(
    git_client=MockGitClient(),
    spec_index=SpecIndex(),
    llm_client=MockLLMClient(),
    language="ru",
)

# Запуск аудита PR или коммита
report = orchestrator.run_audit(
    repo="octocat/Hello-World",
    target_ref="pr/42",
    user_summary="Добавлена валидация длины пароля",
)
print(f"Audit ID: {report.audit_id}, Находок: {len(report.findings)}")
for finding in report.findings:
    print(f"[{finding.status.value}] {finding.req_id}: {finding.title}")

# Оспаривание замечания
from specguard.models.schema import DisputeRequest

dispute_resp = orchestrator.review_dispute(
    dispute=DisputeRequest(
        audit_id=report.audit_id,
        finding_req_id=report.findings[0].req_id,
        user_comment="Проверка вынесена в middleware, требование соблюдено",
    ),
    saved_report=report,
)
print(f"Статус оспаривания: {dispute_resp.dispute_status}")
```

### Модуль 2 (RAG) — Реализовано

Модуль RAG полностью завершен и покрыт 68 тестами:

**Возможности:**
- Парсинг документов (TXT, Markdown, DOCX, PDF)
- Извлечение требований с ID, условиями, исключениями и перекрестными ссылками
- Гибридный поиск: BM25 (лексический) + векторы (семантический) + RRF
- Раскрытие графа связанных требований
- Персистентное хранилище снимков для истории и оспаривания

**Архитектура:**
```
RAGService (фасад)
  ├── parse_document() → RequirementExtractor
  ├── HybridSpecIndex (BM25 + векторы + RRF)
  │   ├── OpenAIEmbeddingClient (Ollama, OpenAI)
  │   └── MockEmbeddingClient (тесты)
  └── SpecSnapshotStore (персистентность)
```

**Тесты:**
Полный оффлайн-сьют (674 теста) запускается автономно без внешних ключей и сети:
```bash
pytest tests/ -v
```

## Технические требования

- Python >=3.10
- Зависимости: `pydantic`, `httpx`, `rank-bm25`, `pypdf`, `python-docx`, `numpy`

## Текущий статус

| Модуль | Статус | Прогресс |
|--------|--------|----------|
| Agent Core & Orchestrator | ✅ Готов (VICTORY CONFIRMED) | 100% |
| RAG & Parser | ✅ Готов | 100% |
| GitHub MCP | ⏳ В разработке | 0% |
| CLI | ⏳ В разработке | 0% |
| Evaluation | ⏳ В разработке | 0% |
| Presentation | ⏳ В разработке | 0% |

## Документация

- [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — План реализации
- [SPEC_SUMMARY.md](docs/SPEC_SUMMARY.md) — Выжимка ТЗ

## Лицензия

MIT
