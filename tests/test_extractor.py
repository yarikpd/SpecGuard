from pathlib import Path
import pytest

from specguard.models.schema import SpecRequirement
from specguard.rag.extractor import RequirementExtractor


@pytest.fixture
def extractor() -> RequirementExtractor:
    return RequirementExtractor()


def test_extract_explicit_ids(extractor: RequirementExtractor):
    text = """
# Функциональные требования
- FR-001: Пользователь добавляет ТЗ в базу и может перезаписать старые данные.
- NFR-002: Исходный код и ТЗ анализируются как данные согласно FR-001.
- AC-001 [FR-001, NFR-002]: В проверочных примерах загружаются TXT, Markdown, DOCX.
- TEST-001: Создать отдельный учебный GitHub-репозиторий.
"""
    reqs = extractor.extract(text, source_file="spec.md")

    assert len(reqs) == 4
    id_map = {r.req_id: r for r in reqs}

    assert "FR-001" in id_map
    assert "NFR-002" in id_map
    assert "AC-001" in id_map
    assert "TEST-001" in id_map

    # Проверка разделов
    assert id_map["FR-001"].section == "Функциональные требования"
    assert id_map["NFR-002"].section == "Функциональные требования"
    assert id_map["FR-001"].source_file == "spec.md"

    # Проверка перекрестных ссылок (related_req_ids)
    assert id_map["FR-001"].related_req_ids == []
    assert id_map["NFR-002"].related_req_ids == ["FR-001"]
    assert id_map["AC-001"].related_req_ids == ["FR-001", "NFR-002"]

    # Проверка, что raw_text содержит полный исходный контекст
    assert "- FR-001: Пользователь добавляет ТЗ в базу" in id_map["FR-001"].raw_text


def test_extract_conditions_and_exceptions(extractor: RequirementExtractor):
    text = """
# Правила обработки
- FR-101: Пользователь авторизуется, если введен правильный пароль.
- FR-102: В случае сбоя сети система повторяет запрос до 3 раз.
- FR-103: Данные синхронизируются при условии наличия активного соединения.
- FR-104: Доступ открыт для всех пользователей, кроме гостей.
- FR-105: Требование применяется ко всем сервисам, за исключением legacy-модуля.
- FR-106: Ограничение не распространяется на учетные записи администраторов.
- FR-107: Заказ подтверждается при условии оплаты, если адрес указан, кроме отдаленных регионов.
"""
    reqs = extractor.extract(text)
    assert len(reqs) == 7
    id_map = {r.req_id: r for r in reqs}

    # "если..."
    assert any("если" in c.lower() for c in id_map["FR-101"].conditions)
    assert id_map["FR-101"].exceptions == []

    # "в случае..."
    assert any("в случае" in c.lower() for c in id_map["FR-102"].conditions)
    assert id_map["FR-102"].exceptions == []

    # "при условии..."
    assert any("при условии" in c.lower() for c in id_map["FR-103"].conditions)
    assert id_map["FR-103"].exceptions == []

    # "кроме..."
    assert any("кроме" in e.lower() for e in id_map["FR-104"].exceptions)
    assert id_map["FR-104"].conditions == []

    # "за исключением..."
    assert any("за исключением" in e.lower() for e in id_map["FR-105"].exceptions)
    assert id_map["FR-105"].conditions == []

    # "не распространяется..."
    assert any("не распространяется" in e.lower() for e in id_map["FR-106"].exceptions)
    assert id_map["FR-106"].conditions == []

    # Множественные условия и исключения в одном требовании
    r107 = id_map["FR-107"]
    assert len(r107.conditions) >= 2
    assert any("при условии" in c.lower() for c in r107.conditions)
    assert any("если" in c.lower() for c in r107.conditions)
    assert any("кроме" in e.lower() for e in r107.exceptions)

    # Исходный raw_text не разорван
    assert "при условии оплаты" in r107.raw_text
    assert "кроме отдаленных регионов" in r107.raw_text


def test_section_hierarchy(extractor: RequirementExtractor):
    text = """
# Архитектура
## Безопасность
- SEC-001: Шифрование трафика в канале.

### Аутентификация
- SEC-002: Поддержка протокола OAuth 2.0.

### Авторизация
- SEC-003: Ролевая модель доступа RBAC.

## Производительность
- PERF-001: Время отклика API не более 200 мс.

# Эксплуатация
- OPS-001: Логирование в формате JSON.
"""
    reqs = extractor.extract(text)
    id_map = {r.req_id: r for r in reqs}

    assert id_map["SEC-001"].section == "Архитектура > Безопасность"
    assert id_map["SEC-002"].section == "Архитектура > Безопасность > Аутентификация"
    assert id_map["SEC-003"].section == "Архитектура > Безопасность > Авторизация"
    assert id_map["PERF-001"].section == "Архитектура > Производительность"
    assert id_map["OPS-001"].section == "Эксплуатация"


def test_fallback_paragraphs_without_ids(extractor: RequirementExtractor):
    text = """
# Общие положения
Система автоматически проверяет код на соответствие заданным правилам.
Проверка запускается пользователем вручную через интерфейс командной строки.

Каждый отчет сохраняется в локальной базе данных, если включен флаг аудита.

Архивные записи не удаляются, кроме случаев явной команды очистки.
"""
    reqs = extractor.extract(text, source_file="general.txt")

    assert len(reqs) == 3
    # Проверка генерации системных ID REQ-001...
    assert [r.req_id for r in reqs] == ["REQ-001", "REQ-002", "REQ-003"]

    for r in reqs:
        assert r.section == "Общие положения"
        assert r.source_file == "general.txt"
        assert r.raw_text

    # Проверка условий и исключений для сгенерированных требований
    assert any("если" in c.lower() for c in reqs[1].conditions)
    assert any("кроме" in e.lower() for e in reqs[2].exceptions)


def test_extract_real_technical_spec(extractor: RequirementExtractor):
    spec_path = Path("specification/TECHNICAL_SPEC.md")
    assert spec_path.exists(), "Файл specification/TECHNICAL_SPEC.md должен существовать"

    text = spec_path.read_text(encoding="utf-8")
    reqs = extractor.extract(text, source_file=str(spec_path))

    # В ТЗ: 13 функциональных (FR), 5 нефункциональных (NFR), 10 тестовых (TEST) = 28
    assert len(reqs) >= 28

    id_map = {r.req_id: r for r in reqs}

    # Проверка наличия ключевых требований
    assert "FR-001" in id_map
    assert "FR-004" in id_map
    assert "NFR-001" in id_map
    assert "TEST-001" in id_map

    # Проверка соответствия разделов
    assert id_map["FR-001"].section == "Техническое задание — утверждено > Функциональные требования"
    assert id_map["NFR-001"].section == "Техническое задание — утверждено > Нефункциональные требования и архитектура"
    assert id_map["TEST-001"].section == "Техническое задание — утверждено > Тестирование на отдельном репозитории"

    # Проверка извлечения условий в реальном ТЗ (NFR-001: "..., если известны.")
    nfr_001 = id_map["NFR-001"]
    assert any("если" in c.lower() for c in nfr_001.conditions)


def test_cross_references_range(extractor: RequirementExtractor):
    text = """
# Приемка
- AC-013 [TEST-001–TEST-005]: Подготовлены тестовый репозиторий и набор эталонов.
"""
    reqs = extractor.extract(text)
    assert len(reqs) == 1
    ac13 = reqs[0]
    assert ac13.req_id == "AC-013"
    assert ac13.related_req_ids == [
        "TEST-001",
        "TEST-002",
        "TEST-003",
        "TEST-004",
        "TEST-005",
    ]


def test_english_predicates(extractor: RequirementExtractor):
    text = """
# API Specs
- FR-201: The system shall refresh tokens when expired, except for revoked clients.
- FR-202: Access is granted provided that API key is valid, unless rate limit exceeded.
"""
    reqs = extractor.extract(text)
    id_map = {r.req_id: r for r in reqs}

    assert any("when" in c.lower() for c in id_map["FR-201"].conditions)
    assert any("except" in e.lower() for e in id_map["FR-201"].exceptions)

    assert any("provided that" in c.lower() for c in id_map["FR-202"].conditions)
    assert any("unless" in e.lower() for e in id_map["FR-202"].exceptions)


def test_empty_text(extractor: RequirementExtractor):
    assert extractor.extract("") == []
    assert extractor.extract("   \n\n   ") == []
