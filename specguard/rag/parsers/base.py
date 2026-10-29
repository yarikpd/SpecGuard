from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentParser(Protocol):
    """Базовый протокол для парсеров документов различных форматов."""

    def parse(self, path: Path) -> tuple[str, list[str]]:
        """Извлекает сырой текст из документа.

        Args:
            path: Путь к файлу документа.

        Returns:
            Кортеж (raw_text, warnings).
        """
        ...
