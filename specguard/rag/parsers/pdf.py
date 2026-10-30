import logging
import threading
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class _FontWarningCollector(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread = threading.get_ident()
        self.count = 0

    def filter(self, record):
        if threading.get_ident() == self.thread and str(record.msg).startswith("Got invalid hex string:"):
            self.count += 1
            return False
        return True


class PdfParser:
    """Парсер PDF-документов с постраничным контролем извлечения текста."""

    def parse(self, path: Path) -> tuple[str, list[str]]:
        collector = _FontWarningCollector()
        logger = logging.getLogger("pypdf._cmap")
        logger.addFilter(collector)
        try:
            text, warnings = self._parse(path)
        finally:
            logger.removeFilter(collector)
        if collector.count:
            warnings.append(
                f"PDF {path.name}: обнаружены некорректные коды символов в таблице шрифта "
                f"({collector.count} сообщений). Часть символов могла не извлечься; "
                "проверьте текст ТЗ или загрузите его в TXT, Markdown либо DOCX."
            )
        return text, warnings

    def _parse(self, path: Path) -> tuple[str, list[str]]:
        if not path.exists():
            return "", [f"Файл не найден: {path}"]
        if not path.is_file():
            return "", [f"Указанный путь не является файлом: {path}"]

        try:
            reader = PdfReader(str(path))
        except (PdfReadError, Exception) as exc:
            return "", [f"Ошибка чтения PDF файла {path.name}: {exc}"]

        if not reader.pages:
            return "", [f"PDF документ {path.name} не содержит страниц."]

        pages_text: list[str] = []
        empty_pages: list[int] = []
        warnings: list[str] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = (page.extract_text() or "").strip()
            except Exception as exc:
                empty_pages.append(page_idx)
                warnings.append(f"Ошибка извлечения текста со страницы {page_idx} ({path.name}): {exc}")
                continue

            if not page_text:
                empty_pages.append(page_idx)
                continue

            pages_text.append(page_text)

        if empty_pages:
            pages_list = ", ".join(map(str, empty_pages))
            warnings.append(
                f"Страницы [{pages_list}] в {path.name} пусты или являются сканами (текст не извлечен)."
            )

        raw_text = "\n\n".join(pages_text).strip()
        if not raw_text:
            warnings.append(f"Файл {path.name} не содержит извлекаемого текста.")

        return raw_text, warnings
