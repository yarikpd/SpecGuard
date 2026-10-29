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

- **Модуль 1** (`specguard/agent/`) — Ядро агента и оркестрация
- **Модуль 2** (`specguard/rag/`) — ✅ RAG & граф требований (100% готов)
- **Модуль 3** (`specguard/mcp/`) — GitHub MCP адаптер
- **Модуль 4** (`specguard/cli/`) — CLI интерфейс
- **Модуль 5** (`specguard/eval/`) — Эталонный датасет и метрики
- **Модуль 6** (`presentation/`) — Презентация и демо

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
```bash
pytest tests/ -v
```

## Технические требования

- Python >=3.10
- Зависимости: `pydantic`, `httpx`, `rank-bm25`, `pypdf`, `python-docx`, `numpy`

## Текущий статус

| Модуль | Статус | Прогресс |
|--------|--------|----------|
| RAG & Parser | ✅ Готов | 100% |
| Agent Core | ⏳ В разработке | 0% |
| GitHub MCP | ⏳ В разработке | 0% |
| CLI | ⏳ В разработке | 0% |
| Evaluation | ⏳ В разработке | 0% |
| Presentation | ⏳ В разработке | 0% |

## Документация

- [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — План реализации
- [SPEC_SUMMARY.md](docs/SPEC_SUMMARY.md) — Выжимка ТЗ

## Лицензия

MIT
