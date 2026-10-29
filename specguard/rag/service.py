from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

from specguard.models.schema import SpecRequirement
from specguard.rag.embeddings import BaseEmbeddingClient
from specguard.rag.extractor import RequirementExtractor
from specguard.rag.index import HybridSpecIndex
from specguard.rag.parsers import parse_document
from specguard.rag.snapshots import SnapshotNotFoundError, SpecSnapshot, SpecSnapshotStore

logger = logging.getLogger(__name__)


def _resolve_model_name(embedding_client: BaseEmbeddingClient) -> str:
    if hasattr(embedding_client, "model_name") and embedding_client.model_name:
        return str(embedding_client.model_name)
    if hasattr(embedding_client, "model") and embedding_client.model:
        return str(embedding_client.model)
    return embedding_client.__class__.__name__


class RAGService:
    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        storage_dir: Path | str = ".specguard_snapshots",
    ):
        if embedding_client is None:
            raise ValueError("embedding_client must be provided")

        self.embedding_client: BaseEmbeddingClient = embedding_client
        self.model_name: str = _resolve_model_name(embedding_client)
        self.snapshot_store: SpecSnapshotStore = SpecSnapshotStore(storage_dir=storage_dir)

    @property
    def storage_dir(self) -> Path:
        return self.snapshot_store.storage_dir

    def index_document(
        self,
        file_path: str | Path,
    ) -> Tuple[HybridSpecIndex, SpecSnapshot, List[str]]:
        path = Path(file_path)
        raw_text, warnings = parse_document(path)
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        cached_snapshot = self.snapshot_store.find_by_content_hash(
            content_hash=content_hash,
            embedding_model=self.model_name,
        )

        if cached_snapshot is not None:
            logger.info(
                "Найден кэшированный снимок ТЗ: snapshot_id=%s (файл: %s)",
                cached_snapshot.snapshot_id,
                path.name,
            )
            snapshot = cached_snapshot
            requirements = snapshot.requirements
            all_warnings = list(snapshot.warnings)
        else:
            if raw_text.strip():
                extractor = RequirementExtractor()
                requirements = extractor.extract(raw_text, source_file=path.name)
            else:
                requirements = []

            snapshot = SpecSnapshot.create(
                source_filename=path.name,
                raw_text=raw_text,
                embedding_model=self.model_name,
                requirements=requirements,
                warnings=warnings,
            )
            self.snapshot_store.save_snapshot(snapshot)
            logger.info(
                "Создан и сохранен новый снимок ТЗ: snapshot_id=%s (требований: %d)",
                snapshot.snapshot_id,
                len(requirements),
            )
            all_warnings = warnings

        index = HybridSpecIndex(self.embedding_client)
        index.build_index(requirements)

        return index, snapshot, all_warnings

    def restore_index(
        self,
        snapshot_id: str,
    ) -> Tuple[HybridSpecIndex, SpecSnapshot]:
        snapshot = self.snapshot_store.load_snapshot(snapshot_id)
        if snapshot is None:
            raise SnapshotNotFoundError(f"Snapshot with ID '{snapshot_id}' not found in storage.")

        if snapshot.embedding_model != self.model_name:
            logger.warning(
                "Восстановление индекса с отличающейся моделью эмбеддингов: "
                "снимок был создан для '%s', текущий клиент '%s'",
                snapshot.embedding_model,
                self.model_name,
            )

        index = HybridSpecIndex(self.embedding_client)
        index.build_index(snapshot.requirements)
        return index, snapshot
