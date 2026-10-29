"""Document parser for TXT, Markdown, DOCX, and PDF (Module 2)."""

import os
from pathlib import Path
from typing import List, Tuple
from specguard.models.schema import SpecRequirement


class SpecParser:
    """Парсер технических заданий различных форматов (FR-002)."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}

    def parse_file(self, file_path: str) -> Tuple[List[SpecRequirement], List[str]]:
        """
        Парсит файл ТЗ и возвращает:
        (список требований, список предупреждений/ограничений).
        """
        path = Path(file_path)
        warnings: List[str] = []

        if not path.exists():
            return [], [f"Файл не найден: {file_path}"]

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return [], [f"Неподдерживаемый формат: {ext}. Поддерживаются TXT, MD, DOCX, PDF."]

        raw_text = ""
        try:
            if ext in {".txt", ".md"}:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()

            elif ext == ".docx":
                try:
                    import docx
                    doc = docx.Document(file_path)
                    raw_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                except ImportError:
                    warnings.append("Библиотека python-docx не установлена, извлечение DOCX ограничено.")
                    return [], warnings

            elif ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    pages_text = []
                    for idx, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        if not page_text.strip():
                            warnings.append(f"Страница {idx+1} в PDF пуста или содержит нераспознанный скан.")
                        pages_text.append(page_text)
                    raw_text = "\n".join(pages_text)
                except ImportError:
                    warnings.append("Библиотека pypdf не установлена, извлечение PDF ограничено.")
                    return [], warnings

        except Exception as e:
            warnings.append(f"Ошибка при чтении файла {file_path}: {str(e)}")
            return [], warnings

        if not raw_text.strip():
            warnings.append(f"Файл {file_path} не содержит извлекаемого текста.")
            return [], warnings

        requirements = self._extract_requirements(raw_text, source_file=path.name)
        return requirements, warnings

    def _extract_requirements(self, text: str, source_file: str) -> List[SpecRequirement]:
        """Базовое разбиение текста на требования по идентификаторам и параграфам."""
        lines = text.split("\n")
        reqs: List[SpecRequirement] = []
        current_section = "General"
        req_counter = 1

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("#"):
                current_section = line_str.lstrip("#").strip()
                continue

            # Ищем маркеры вида REQ-001, FR-001, или маркеры списка
            req_id = f"REQ-{req_counter:03d}"
            if line_str.startswith(("- ", "* ")):
                content = line_str[2:].strip()
                # Если в начале уже есть идентификатор (например FR-001:)
                if ":" in content and len(content.split(":")[0]) < 12:
                    potential_id, rest = content.split(":", 1)
                    req_id = potential_id.strip()
                    content = rest.strip()

                reqs.append(SpecRequirement(
                    req_id=req_id,
                    section=current_section,
                    title=content[:50] + ("..." if len(content) > 50 else ""),
                    description=content,
                    raw_text=line_str,
                    source_file=source_file
                ))
                req_counter += 1

        # Если явных пунктов списка не нашлось, разобьем по параграфам
        if not reqs and text.strip():
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                reqs.append(SpecRequirement(
                    req_id=f"REQ-{req_counter:03d}",
                    section=current_section,
                    title=p[:40] + "...",
                    description=p,
                    raw_text=p,
                    source_file=source_file
                ))
                req_counter += 1

        return reqs
