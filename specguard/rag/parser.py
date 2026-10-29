from pathlib import Path
from typing import List, Tuple

from specguard.models.schema import SpecRequirement
from specguard.rag.parsers import parse_document


class SpecParser:
    def parse_file(self, file_path: str) -> Tuple[List[SpecRequirement], List[str]]:
        path = Path(file_path)
        raw_text, warnings = parse_document(path)

        if not raw_text.strip():
            return [], warnings

        requirements = self._extract_requirements(raw_text, source_file=path.name)
        return requirements, warnings

    def _extract_requirements(self, text: str, source_file: str) -> List[SpecRequirement]:
        from specguard.rag.extractor import RequirementExtractor
        return RequirementExtractor().extract(text, source_file=source_file)
