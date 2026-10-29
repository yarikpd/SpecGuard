# SpecGuard 🛡️

> Автономный ИИ-агент для автоматического аудита соответствия Pull Request и коммитов в GitHub требованиям ТЗ (через RAG и GitHub MCP).

Проект подготовлен для кейс-чемпионата.

---

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Запуск сквозного аудита (в Mock-режиме)
```bash
python -m specguard.cli.main audit --spec "specification/TECHNICAL_SPEC.md" --repo "my-org/my-app" --target "42" --summary "Добавлена валидация длины пароля"
```

### 3. Оспаривание замечания (Re-evaluation)
```bash
python -m specguard.cli.main dispute <audit_id> --req "REQ-001" --comment "В строке 15 есть исключение из правила"
```

### 4. Запуск эталонного бенчмарка
```bash
python -m specguard.eval.benchmark_runner
```

---

## 🏛️ Архитектура системы

- **Модуль 1 (`specguard/agent/`):** Ядро агента и оркестрация сопоставления ТЗ и кода.
- **Модуль 2 (`specguard/rag/`):** Парсер 4 форматов (TXT/MD/DOCX/PDF) и гибридный поиск требований.
- **Модуль 3 (`specguard/mcp/`):** Адаптер к GitHub MCP Server (сравнение PR, коммитов и merge-коммитов по 1-му родителю).
- **Модуль 4 (`specguard/cli/`):** CLI интерфейс на Typer & Rich, независимые профили моделей, история проверок и оспаривание.
- **Модуль 5 (`specguard/eval/`):** Эталонный датасет (Ground Truth) и расчет метрик качества.
- **Модуль 6 (`presentation/`):** Слайды презентации, скринкаст-демо и финальный питч.

---

## 📋 План проекта и критерии сдачи
Подробный почасовой план распределения задач на 6 человек находится в [PROJECT_PLAN.md](PROJECT_PLAN.md).  
Выжимка технического задания находится в [docs/SPEC_SUMMARY.md](docs/SPEC_SUMMARY.md).
