from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from specguard.models.schema import SpecRequirement
from specguard.rag.embeddings import (
    BaseEmbeddingClient,
    EmbeddingError,
    MockEmbeddingClient,
)

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[\w]+(?:-[\w]+)*")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def requirement_to_text(req: SpecRequirement) -> str:
    parts: list[str] = []
    if req.req_id:
        parts.append(req.req_id)
        parts.append(f"Идентификатор: {req.req_id}")
    if req.section:
        parts.append(f"Раздел: {req.section}")
    if req.title and req.title != req.description:
        parts.append(f"Название: {req.title}")
    if req.description:
        parts.append(f"Описание: {req.description}")
    elif req.raw_text:
        parts.append(f"Текст: {req.raw_text}")
    if req.conditions:
        parts.append("Условия: " + "; ".join(req.conditions))
    if req.exceptions:
        parts.append("Исключения: " + "; ".join(req.exceptions))
    if req.related_req_ids:
        parts.append("Связанные требования: " + " ".join(req.related_req_ids))
    return "\n".join(parts)


class SearchResults(list):
    def __init__(
        self,
        items: Optional[List[SpecRequirement]] = None,
        is_fallback: bool = False,
    ):
        super().__init__(items or [])
        self.is_fallback: bool = is_fallback

    @property
    def fallback(self) -> bool:
        return self.is_fallback


class HybridSpecIndex:
    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        rrf_k: int = 60,
        min_semantic_score: float = 0.15,
    ):
        if embedding_client is None:
            raise ValueError("embedding_client must be provided")
        self.embedding_client: BaseEmbeddingClient = embedding_client
        self.rrf_k: int = rrf_k
        self.min_semantic_score: float = min_semantic_score

        self.requirements: list[SpecRequirement] = []
        self._id_map: dict[str, SpecRequirement] = {}
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None
        self.last_search_is_fallback: bool = False

    @property
    def is_fallback(self) -> bool:
        return self.last_search_is_fallback

    def get_by_id(self, req_id: str) -> SpecRequirement | None:
        if not req_id:
            return None
        return self._id_map.get(req_id.strip().upper())

    def build_index(self, requirements: list[SpecRequirement]) -> None:
        self.requirements = list(requirements)
        self._id_map = {
            r.req_id.strip().upper(): r
            for r in self.requirements
            if r.req_id
        }
        self.last_search_is_fallback = False

        if not self.requirements:
            self.bm25 = None
            self.embeddings = np.empty((0, self.embedding_client.dimension), dtype=np.float32)
            return

        corpus_texts = [requirement_to_text(r) for r in self.requirements]

        tokenized_corpus = [tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # fix for small corpus where BM25 can produce negative IDF
        for term, idf_val in self.bm25.idf.items():
            if idf_val <= 0:
                self.bm25.idf[term] = 0.25

        raw_embeddings = self.embedding_client.embed_batch(corpus_texts)
        if raw_embeddings:
            emb_matrix = np.array(raw_embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = emb_matrix / norms
        else:
            self.embeddings = np.empty((0, self.embedding_client.dimension), dtype=np.float32)

    def set_requirements(self, requirements: list[SpecRequirement]) -> None:
        self.build_index(requirements)

    def search(
        self,
        query: str,
        top_k: int = 5,
        expand_related: bool = True,
        min_semantic_score: float | None = None,
    ) -> list[SpecRequirement]:
        n = len(self.requirements)
        if n == 0:
            self.last_search_is_fallback = True
            return SearchResults([], is_fallback=True)

        query_clean = query.strip()
        if not query_clean:
            self.last_search_is_fallback = True
            return SearchResults(self.requirements[:top_k], is_fallback=True)

        query_tokens = tokenize(query_clean)
        rrf_scores = np.zeros(n, dtype=np.float64)

        bm25_scores = np.zeros(n, dtype=np.float32)
        has_bm25_hits = False
        if self.bm25 is not None and query_tokens:
            bm25_scores = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)
            pos_bm25_indices = [i for i in range(n) if bm25_scores[i] > 0]
            if pos_bm25_indices:
                has_bm25_hits = True
                pos_bm25_indices.sort(key=lambda i: bm25_scores[i], reverse=True)
                for rank, idx in enumerate(pos_bm25_indices, start=1):
                    rrf_scores[idx] += 1.0 / (self.rrf_k + rank)

        sem_scores = np.zeros(n, dtype=np.float32)
        has_sem_hits = False
        if self.embeddings is not None and len(self.embeddings) == n:
            try:
                raw_q_vec = self.embedding_client.embed_text(query_clean)
                q_vec = np.array(raw_q_vec, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 1e-12:
                    q_vec = q_vec / q_norm
                sem_scores = np.dot(self.embeddings, q_vec)

                effective_threshold = (
                    self.min_semantic_score
                    if min_semantic_score is None
                    else min_semantic_score
                )
                pos_sem_indices = [i for i in range(n) if sem_scores[i] > effective_threshold]
                if pos_sem_indices:
                    has_sem_hits = True
                    pos_sem_indices.sort(key=lambda i: sem_scores[i], reverse=True)
                    for rank, idx in enumerate(pos_sem_indices, start=1):
                        rrf_scores[idx] += 1.0 / (self.rrf_k + rank)
            except EmbeddingError as exc:
                logger.warning("Semantic embedding query failed, falling back to lexical: %s", exc)

        if not has_bm25_hits and not has_sem_hits:
            self.last_search_is_fallback = True
            return SearchResults(self.requirements[:top_k], is_fallback=True)

        candidate_indices = [i for i in range(n) if rrf_scores[i] > 0]
        candidate_indices.sort(
            key=lambda i: (rrf_scores[i], bm25_scores[i], sem_scores[i]),
            reverse=True,
        )

        selected_indices = candidate_indices[:top_k]
        results = [self.requirements[i] for i in selected_indices]

        if expand_related:
            seen_ids = {r.req_id.upper() for r in results if r.req_id}
            expanded = list(results)
            for r in results:
                for rel_id in r.related_req_ids:
                    clean_rel_id = rel_id.strip().upper()
                    if clean_rel_id and clean_rel_id not in seen_ids:
                        rel_req = self.get_by_id(clean_rel_id)
                        if rel_req is not None:
                            seen_ids.add(clean_rel_id)
                            expanded.append(rel_req)
            results = expanded

        self.last_search_is_fallback = False
        return SearchResults(results, is_fallback=False)


class SpecIndex(HybridSpecIndex):
    def __init__(
        self,
        requirements: Optional[List[SpecRequirement]] = None,
        embedding_client: Optional[BaseEmbeddingClient] = None,
        rrf_k: int = 60,
        min_semantic_score: float = 0.15,
    ):
        client = (
            embedding_client
            if embedding_client is not None
            else MockEmbeddingClient()
        )
        super().__init__(
            embedding_client=client,
            rrf_k=rrf_k,
            min_semantic_score=min_semantic_score,
        )
        if requirements:
            self.build_index(requirements)
