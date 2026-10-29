"""Tests for SpecSnapshot, SpecSnapshotStore and RAGService (FR-010, FR-011, FR-012)."""

from pathlib import Path
from unittest.mock import patch
import pytest

from specguard.models.schema import SpecRequirement
from specguard.rag import (
    BaseEmbeddingClient,
    HybridSpecIndex,
    MockEmbeddingClient,
    RAGService,
    SnapshotNotFoundError,
    SpecSnapshot,
    SpecSnapshotStore,
)


SAMPLE_SPEC_V1 = """# Техническое задание (Версия 1.0)

## Раздел 1. Аутентификация
- FR-001: Пользователь должен иметь возможность авторизоваться в системе по email и паролю.
  Условие: при наличии активной учетной записи.
  Исключение: кроме заблокированных пользователей.
- FR-002: Длина пароля должна быть не менее 8 символов.
  Связанные требования: FR-001.
"""

SAMPLE_SPEC_V2 = """# Техническое задание (Версия 2.0)

## Раздел 1. Аутентификация
- FR-001: Пользователь должен иметь возможность авторизоваться в системе по email и паролю с поддержкой 2FA.
  Условие: при наличии подтвержденного номера телефона или TOTP.
  Исключение: кроме экстренных сервисных аккаунтов.
- FR-002: Длина пароля должна быть не менее 12 символов.
- FR-003: При 5 неудачных попытках входа аккаунт временно блокируется на 15 минут.
"""


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    """Временная директория для хранения снимков."""
    dir_path = tmp_path / ".specguard_snapshots"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture
def mock_embedding_client() -> MockEmbeddingClient:
    return MockEmbeddingClient(dimension=64, model_name="mock-embeddings-v1")


@pytest.fixture
def rag_service(mock_embedding_client: MockEmbeddingClient, snapshot_dir: Path) -> RAGService:
    return RAGService(embedding_client=mock_embedding_client, storage_dir=snapshot_dir)


class TestSpecSnapshotAndStore:
    """Тестирование модели снимка и хранилища SpecSnapshotStore."""

    def test_save_and_load_snapshot_disk_persistence(self, snapshot_dir: Path):
        """Тест сохранения и загрузки снимка с диска (FR-010)."""
        store = SpecSnapshotStore(storage_dir=snapshot_dir)

        req1 = SpecRequirement(
            req_id="FR-001",
            section="Безопасность",
            title="Авторизация",
            description="Требование авторизации пользователя",
            conditions=["при активном токене"],
            exceptions=["кроме суперпользователя"],
            raw_text="FR-001: Авторизация",
            source_file="spec.md",
            related_req_ids=["FR-002"],
        )
        req2 = SpecRequirement(
            req_id="FR-002",
            section="Безопасность",
            title="Сложность пароля",
            description="Пароль минимум 8 символов",
            source_file="spec.md",
        )

        snapshot = SpecSnapshot.create(
            source_filename="spec.md",
            raw_text=SAMPLE_SPEC_V1,
            embedding_model="mock-embeddings-v1",
            requirements=[req1, req2],
            warnings=["Предупреждение парсера 1"],
        )

        # Сохранение на диск
        snapshot_id = store.save_snapshot(snapshot)
        assert snapshot_id == snapshot.snapshot_id
        assert snapshot_id is not None and len(snapshot_id) == 64

        # Проверяем, что на диске действительно создан JSON-файл
        expected_json_file = snapshot_dir / f"{snapshot_id}.json"
        assert expected_json_file.is_file()

        # Создаем НОВЫЙ экземпляр хранилища с пустой памятью, чтобы проверить чтение с диска
        store_from_disk = SpecSnapshotStore(storage_dir=snapshot_dir)
        loaded = store_from_disk.load_snapshot(snapshot_id)

        assert loaded is not None
        assert loaded.snapshot_id == snapshot.snapshot_id
        assert loaded.source_filename == "spec.md"
        assert loaded.content_hash == snapshot.content_hash
        assert loaded.embedding_model == "mock-embeddings-v1"
        assert loaded.created_at == snapshot.created_at
        assert loaded.warnings == ["Предупреждение парсера 1"]

        # Проверяем целостность требований
        assert len(loaded.requirements) == 2
        loaded_req1 = loaded.requirements[0]
        assert loaded_req1.req_id == "FR-001"
        assert loaded_req1.section == "Безопасность"
        assert loaded_req1.description == "Требование авторизации пользователя"
        assert loaded_req1.conditions == ["при активном токене"]
        assert loaded_req1.exceptions == ["кроме суперпользователя"]
        assert loaded_req1.related_req_ids == ["FR-002"]

        loaded_req2 = loaded.requirements[1]
        assert loaded_req2.req_id == "FR-002"

    def test_load_snapshot_non_existent_returns_none(self, snapshot_dir: Path):
        """Загрузка несуществующего snapshot_id возвращает None."""
        store = SpecSnapshotStore(storage_dir=snapshot_dir)
        assert store.load_snapshot("missing_snapshot_12345") is None

    def test_find_by_content_hash_caching(self, snapshot_dir: Path):
        """Тест поиска снимка по content_hash и модели эмбеддингов."""
        store = SpecSnapshotStore(storage_dir=snapshot_dir)
        snap = SpecSnapshot.create(
            source_filename="spec.txt",
            raw_text="Уникальный текст технического задания",
            embedding_model="model-alpha",
            requirements=[],
        )
        store.save_snapshot(snap)

        # 1. Точное совпадение
        found = store.find_by_content_hash(snap.content_hash, "model-alpha")
        assert found is not None
        assert found.snapshot_id == snap.snapshot_id

        # 2. Несовпадение content_hash
        assert store.find_by_content_hash("wrong_hash", "model-alpha") is None

        # 3. Несовпадение embedding_model
        assert store.find_by_content_hash(snap.content_hash, "model-beta") is None


class TestRAGService:
    """Тестирование единого фасада RAGService."""

    def test_index_document_creates_snapshot_and_hybrid_index(
        self,
        rag_service: RAGService,
        tmp_path: Path,
    ):
        """Базовый тест индексации документа ТЗ: создается индекс и снимок."""
        spec_path = tmp_path / "spec.md"
        spec_path.write_text(SAMPLE_SPEC_V1, encoding="utf-8")

        index, snapshot, warnings = rag_service.index_document(spec_path)

        assert isinstance(index, HybridSpecIndex)
        assert isinstance(snapshot, SpecSnapshot)
        assert snapshot.source_filename == "spec.md"
        assert len(snapshot.requirements) >= 2
        assert index.get_by_id("FR-001") is not None
        assert index.get_by_id("FR-002") is not None

        # Проверяем успешный поиск
        results = index.search("пароль")
        assert len(results) > 0
        found_ids = [r.req_id for r in results]
        assert "FR-002" in found_ids

    def test_caching_avoids_reparsing_on_identical_document(
        self,
        rag_service: RAGService,
        tmp_path: Path,
    ):
        """Тест кэширования: повторный вызов index_document для неизменного файла

        возвращает кэшированный снимок без повторного парсинга требований.
        """
        spec_path = tmp_path / "requirements.md"
        spec_path.write_text(SAMPLE_SPEC_V1, encoding="utf-8")

        # Первый вызов: документ парсится, создается снимок
        index1, snap1, warns1 = rag_service.index_document(spec_path)
        assert snap1.snapshot_id is not None

        # Второй вызов для того же файла: проверяем, что RequirementExtractor.extract не вызывается
        with patch("specguard.rag.service.RequirementExtractor.extract") as mock_extract:
            index2, snap2, warns2 = rag_service.index_document(spec_path)
            mock_extract.assert_not_called()

        # Снимки должны полностью совпадать
        assert snap1.snapshot_id == snap2.snapshot_id
        assert snap1.content_hash == snap2.content_hash
        assert snap1.created_at == snap2.created_at
        assert len(snap2.requirements) == len(snap1.requirements)

        # Индекс работоспособен
        results = index2.search("авторизоваться")
        assert len(results) > 0
        assert results[0].req_id in ["FR-001", "FR-002"]

    def test_version_isolation_on_document_update_fr010(
        self,
        rag_service: RAGService,
        tmp_path: Path,
    ):
        """Тест изоляции версий (FR-010): изменение файла ТЗ создает НОВЫЙ снимок с новым snapshot_id,

        а старый снимок остается неизменным и доступным.
        """
        spec_path = tmp_path / "spec_active.md"

        # 1. Индексация версии 1
        spec_path.write_text(SAMPLE_SPEC_V1, encoding="utf-8")
        index_v1, snap_v1, warns_v1 = rag_service.index_document(spec_path)
        id_v1 = snap_v1.snapshot_id
        assert len(snap_v1.requirements) == 2

        # 2. Обновление (перезапись) того же файла на версию 2
        spec_path.write_text(SAMPLE_SPEC_V2, encoding="utf-8")
        index_v2, snap_v2, warns_v2 = rag_service.index_document(spec_path)
        id_v2 = snap_v2.snapshot_id
        assert len(snap_v2.requirements) == 3

        # Проверка изоляции
        assert id_v1 != id_v2, "Идентификаторы снимков разных версий ТЗ должны отличаться"
        assert snap_v1.content_hash != snap_v2.content_hash

        # 3. Убеждаемся, что старый снимок v1 остался на диске неизменным
        loaded_v1 = rag_service.snapshot_store.load_snapshot(id_v1)
        assert loaded_v1 is not None
        assert loaded_v1.snapshot_id == id_v1
        assert len(loaded_v1.requirements) == 2
        req1_v1 = next(r for r in loaded_v1.requirements if r.req_id == "FR-001")
        assert "2FA" not in req1_v1.description

        # 4. Проверяем новый снимок v2
        loaded_v2 = rag_service.snapshot_store.load_snapshot(id_v2)
        assert loaded_v2 is not None
        assert loaded_v2.snapshot_id == id_v2
        assert len(loaded_v2.requirements) == 3
        req1_v2 = next(r for r in loaded_v2.requirements if r.req_id == "FR-001")
        assert "2FA" in req1_v2.description

    def test_restore_historical_index_for_dispute_fr012(
        self,
        rag_service: RAGService,
        tmp_path: Path,
    ):
        """Тест восстановления исторического индекса по snapshot_id (FR-012).

        Пересмотр замечаний обязан выполняться строго на материалах исходной проверки,
        даже если актуальное ТЗ на диске изменилось или было удалено.
        """
        spec_path = tmp_path / "original_spec.md"
        spec_path.write_text(SAMPLE_SPEC_V1, encoding="utf-8")

        # 1. Первоначальный аудит: строим индекс и получаем снимок
        orig_index, orig_snapshot, _ = rag_service.index_document(spec_path)
        orig_snapshot_id = orig_snapshot.snapshot_id

        # 2. Файл ТЗ на диске удаляется (или заменяется другим содержимым)
        spec_path.unlink()
        assert not spec_path.exists()

        # 3. Восстанавливаем исторический индекс строго по snapshot_id
        restored_index, restored_snapshot = rag_service.restore_index(orig_snapshot_id)

        assert restored_snapshot.snapshot_id == orig_snapshot_id
        assert restored_snapshot.source_filename == "original_spec.md"
        assert len(restored_snapshot.requirements) == 2

        # 4. Успешный поиск по восстановленному индексу
        search_res = restored_index.search("электронная почта и пароль")
        assert len(search_res) > 0
        assert any(r.req_id == "FR-001" for r in search_res)

        # FR-003 не существовало в исходной версии v1
        assert restored_index.get_by_id("FR-003") is None

    def test_restore_index_non_existent_snapshot_raises_error(self, rag_service: RAGService):
        """Попытка восстановить несуществующий снимок вызывает SnapshotNotFoundError."""
        with pytest.raises(SnapshotNotFoundError):
            rag_service.restore_index("non_existent_snapshot_id_xyz")

    def test_embedding_model_profile_change_triggers_new_snapshot(
        self,
        snapshot_dir: Path,
        tmp_path: Path,
    ):
        """Тест RAG_DESIGN.md: смена embedding-профиля требует нового индекса/снимка."""
        spec_path = tmp_path / "same_spec.md"
        spec_path.write_text(SAMPLE_SPEC_V1, encoding="utf-8")

        client_model_a = MockEmbeddingClient(dimension=64, model_name="model-qwen")
        client_model_b = MockEmbeddingClient(dimension=64, model_name="model-bge")

        service_a = RAGService(embedding_client=client_model_a, storage_dir=snapshot_dir)
        service_b = RAGService(embedding_client=client_model_b, storage_dir=snapshot_dir)

        _, snap_a, _ = service_a.index_document(spec_path)
        _, snap_b, _ = service_b.index_document(spec_path)

        assert snap_a.content_hash == snap_b.content_hash
        assert snap_a.snapshot_id != snap_b.snapshot_id
        assert snap_a.embedding_model == "model-qwen"
        assert snap_b.embedding_model == "model-bge"

    def test_empty_document_handling(self, rag_service: RAGService, tmp_path: Path):
        """Тест обработки пустого документа."""
        empty_spec = tmp_path / "empty.md"
        empty_spec.write_text("", encoding="utf-8")

        index, snapshot, warnings = rag_service.index_document(empty_spec)
        assert len(snapshot.requirements) == 0
        assert snapshot.snapshot_id is not None
        assert index.requirements == []


class TestPublicExports:
    """Тест экспортов модуля specguard.rag."""

    def test_exports_available(self):
        import specguard.rag as rag

        assert hasattr(rag, "RAGService")
        assert hasattr(rag, "SpecSnapshotStore")
        assert hasattr(rag, "SpecSnapshot")
        assert hasattr(rag, "HybridSpecIndex")
        assert hasattr(rag, "SpecIndex")
        assert hasattr(rag, "BaseEmbeddingClient")
