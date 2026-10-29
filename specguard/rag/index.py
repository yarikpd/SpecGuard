"""RAG and Requirement Search Engine (Module 2)."""

from typing import List, Optional
from specguard.models.schema import SpecRequirement


class SpecIndex:
    """Индекс требований с гибридным/лексическим поиском (RAG)."""

    def __init__(self, requirements: Optional[List[SpecRequirement]] = None):
        self.requirements: List[SpecRequirement] = requirements or []

    def set_requirements(self, requirements: List[SpecRequirement]):
        self.requirements = requirements

    def search(self, query: str, top_k: int = 5) -> List[SpecRequirement]:
        """Поиск релевантных требований по ключевым словам и семантике."""
        if not self.requirements:
            return []

        # Базовый лексический скоринг по вхождению слов
        query_words = set(query.lower().split())
        scored = []
        for req in self.requirements:
            req_text = f"{req.req_id} {req.title} {req.description} {req.section}".lower()
            matches = sum(1 for w in query_words if len(w) > 2 and w in req_text)
            scored.append((matches, req))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored[:top_k]]
        # Если совпадений мало, возвращаем первые требования в качестве контекста
        if not results:
            results = self.requirements[:top_k]
        return results
