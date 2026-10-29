from pathlib import Path


class TextParser:
    """Парсер текстовых форматов (.txt, .md)."""

    ENCODINGS = ("utf-8", "cp1251", "latin-1")

    def parse(self, path: Path) -> tuple[str, list[str]]:
        if not path.exists():
            return "", [f"Файл не найден: {path}"]
        if not path.is_file():
            return "", [f"Указанный путь не является файлом: {path}"]

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            return "", [f"Ошибка чтения файла {path.name}: {exc}"]

        if not raw_bytes:
            return "", [f"Файл {path.name} пуст."]

        text: str | None = None
        used_encoding: str | None = None

        for encoding in self.ENCODINGS:
            try:
                text = raw_bytes.decode(encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        warnings: list[str] = []

        if text is None:
            text = raw_bytes.decode("utf-8", errors="replace")
            warnings.append(f"Кодировка {path.name} не определена, файл прочитан с подстановкой символов.")
        elif used_encoding != "utf-8":
            warnings.append(f"Файл {path.name} прочитан с использованием fallback-кодировки {used_encoding}.")

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            warnings.append(f"Файл {path.name} не содержит значащего текста.")

        return text, warnings
