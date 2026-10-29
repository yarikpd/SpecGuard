from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfParser:
    """Парсер PDF-документов с постраничным контролем извлечения текста."""

    def parse(self, path: Path) -> tuple[str, list[str]]:
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
