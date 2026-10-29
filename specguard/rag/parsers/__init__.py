from pathlib import Path

from specguard.rag.parsers.base import DocumentParser
from specguard.rag.parsers.docx import DocxParser
from specguard.rag.parsers.pdf import PdfParser
from specguard.rag.parsers.text import TextParser

__all__ = [
    "DocumentParser",
    "TextParser",
    "DocxParser",
    "PdfParser",
    "get_parser",
    "parse_document",
]

PARSER_REGISTRY: dict[str, type[DocumentParser]] = {
    ".txt": TextParser,
    ".md": TextParser,
    ".markdown": TextParser,
    ".docx": DocxParser,
    ".pdf": PdfParser,
}


def get_parser(path: str | Path) -> DocumentParser:
    file_path = Path(path)
    ext = file_path.suffix.lower()
    parser_cls = PARSER_REGISTRY.get(ext)
    if not parser_cls:
        supported = ", ".join(sorted(PARSER_REGISTRY.keys()))
        raise ValueError(f"Неподдерживаемый формат файла: '{ext}'. Поддерживаются: {supported}")
    return parser_cls()


def parse_document(path: str | Path) -> tuple[str, list[str]]:
    file_path = Path(path)
    if not file_path.exists():
        return "", [f"Файл не найден: {file_path}"]
    if not file_path.is_file():
        return "", [f"Указанный путь не является файлом: {file_path}"]

    try:
        parser = get_parser(file_path)
        return parser.parse(file_path)
    except ValueError as exc:
        return "", [str(exc)]
    except Exception as exc:
        return "", [f"Непредвиденная ошибка при парсинге {file_path.name}: {exc}"]
