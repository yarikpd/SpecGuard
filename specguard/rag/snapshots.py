from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from specguard.models.schema import SpecRequirement

logger = logging.getLogger(__name__)


class SnapshotNotFoundError(KeyError, ValueError):
    pass


class SpecSnapshot(BaseModel):
    snapshot_id: str = ""
    source_filename: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_hash: str
    embedding_model: str
    requirements: List[SpecRequirement] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @staticmethod
    def compute_snapshot_id(content_hash: str, embedding_model: str, source_filename: str) -> str:
        seed = f"{content_hash}:{embedding_model}:{source_filename}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _ensure_snapshot_id(self) -> "SpecSnapshot":
        if not self.snapshot_id:
            self.snapshot_id = self.compute_snapshot_id(
                content_hash=self.content_hash,
                embedding_model=self.embedding_model,
                source_filename=self.source_filename,
            )
        return self

    @classmethod
    def create(
        cls,
        source_filename: str,
        raw_text: str,
        embedding_model: str,
        requirements: List[SpecRequirement],
        warnings: Optional[List[str]] = None,
        created_at: Optional[str] = None,
        snapshot_id: Optional[str] = None,
    ) -> "SpecSnapshot":
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if snapshot_id is None:
            snapshot_id = cls.compute_snapshot_id(
                content_hash=content_hash,
                embedding_model=embedding_model,
                source_filename=source_filename,
            )
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        return cls(
            snapshot_id=snapshot_id,
            source_filename=source_filename,
            created_at=created_at,
            content_hash=content_hash,
            embedding_model=embedding_model,
            requirements=requirements,
            warnings=warnings or [],
        )


class SpecSnapshotStore:
    def __init__(self, storage_dir: Path | str = ".specguard_snapshots"):
        self.storage_dir: Path = Path(storage_dir)
        self._cache: dict[str, SpecSnapshot] = {}

    def save_snapshot(self, snapshot: SpecSnapshot) -> str:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.storage_dir / f"{snapshot.snapshot_id}.json"
        temp_file = file_path.with_suffix(".tmp")
        json_data = snapshot.model_dump_json(indent=2)
        temp_file.write_text(json_data, encoding="utf-8")
        temp_file.replace(file_path)

        self._cache[snapshot.snapshot_id] = snapshot
        return snapshot.snapshot_id

    def load_snapshot(self, snapshot_id: str) -> Optional[SpecSnapshot]:
        if not snapshot_id:
            return None

        if snapshot_id in self._cache:
            return self._cache[snapshot_id]

        file_path = self.storage_dir / f"{snapshot_id}.json"
        if not file_path.is_file():
            return None

        try:
            content = file_path.read_text(encoding="utf-8")
            snapshot = SpecSnapshot.model_validate_json(content)
            self._cache[snapshot.snapshot_id] = snapshot
            return snapshot
        except Exception as exc:
            logger.warning("Не удалось прочитать снимок %s: %s", snapshot_id, exc)
            return None

    def find_by_content_hash(
        self,
        content_hash: str,
        embedding_model: str,
    ) -> Optional[SpecSnapshot]:
        for snap in self._cache.values():
            if snap.content_hash == content_hash and snap.embedding_model == embedding_model:
                return snap

        if not self.storage_dir.exists():
            return None

        matching: list[SpecSnapshot] = []
        for json_file in self.storage_dir.glob("*.json"):
            sid = json_file.stem
            if sid in self._cache:
                snap = self._cache[sid]
            else:
                try:
                    content = json_file.read_text(encoding="utf-8")
                    snap = SpecSnapshot.model_validate_json(content)
                    self._cache[snap.snapshot_id] = snap
                except Exception as exc:
                    logger.debug("Пропуск поврежденного файла снимка %s: %s", json_file, exc)
                    continue

            if snap.content_hash == content_hash and snap.embedding_model == embedding_model:
                matching.append(snap)

        if not matching:
            return None

        matching.sort(key=lambda s: s.created_at, reverse=True)
        return matching[0]

    def list_snapshots(self) -> list[SpecSnapshot]:
        if not self.storage_dir.exists():
            return list(self._cache.values())

        for json_file in self.storage_dir.glob("*.json"):
            sid = json_file.stem
            if sid not in self._cache:
                try:
                    content = json_file.read_text(encoding="utf-8")
                    snap = SpecSnapshot.model_validate_json(content)
                    self._cache[snap.snapshot_id] = snap
                except Exception:
                    continue

        result = list(self._cache.values())
        result.sort(key=lambda s: s.created_at, reverse=True)
        return result
