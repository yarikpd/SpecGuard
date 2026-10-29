import pytest
from specguard.models.schema import SpecRequirement
from specguard.rag.embeddings import (
    BaseEmbeddingClient,
    EmbeddingConnectionError,
    MockEmbeddingClient,
    l2_normalize,
)
from specguard.rag.index import (
    HybridSpecIndex,
    SearchResults,
    SpecIndex,
    requirement_to_text,
    tokenize,
)


@pytest.fixture
def sample_requirements() -> list[SpecRequirement]:
    return [
        SpecRequirement(
            req_id="FR-001",
            section="Аутентификация",
            title="JWT токены",
            description="Пользователь проходит проверку через метод authenticate_user(username, password).",
            conditions=["Пользователь активен"],
            exceptions=["Учетная запись заблокирована"],
            related_req_ids=["EXC-001", "FR-002"],
        ),
        SpecRequirement(
            req_id="FR-002",
            section="Биллинг",
            title="Оплата картой",
            description="Списание средств выполняется методом process_payment(amount, currency).",
            conditions=["Баланс положительный"],
            exceptions=["Недостаточно средств"],
            related_req_ids=[],
        ),
        SpecRequirement(
            req_id="AC-003",
            section="Безопасность",
            title="Сложность пароля",
            description="Пароль должен содержать не менее 8 символов и спецсимвол при регистрации.",
            conditions=[],
            exceptions=[],
            related_req_ids=[],
        ),
        SpecRequirement(
            req_id="EXC-001",
            section="Исключения",
            title="Блокировка аккаунта",
            description="При 3 неверных попытках входа учетная запись блокируется на 15 минут.",
            conditions=["3 неудачные попытки"],
            exceptions=[],
            related_req_ids=["FR-001"],
        ),
    ]


class TestTokenizer:
    def test_preserves_identifiers_and_methods(self):
        text = "FR-001: Вызов метода verify_token(user_id) в AC-003. Код 404."
        tokens = tokenize(text)
        assert "fr-001" in tokens
        assert "ac-003" in tokens
        assert "verify_token" in tokens
        assert "user_id" in tokens
        assert "404" in tokens

    def test_empty_string(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestExactMatchBM25:
    def test_find_by_exact_req_id(self, sample_requirements: list[SpecRequirement]):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        # Поиск по идентификатору FR-001
        results = index.search("FR-001", top_k=1, expand_related=False)
        assert len(results) == 1
        assert results[0].req_id == "FR-001"
        assert results.is_fallback is False

        # Поиск по идентификатору AC-003
        results_ac = index.search("AC-003", top_k=1, expand_related=False)
        assert len(results_ac) == 1
        assert results_ac[0].req_id == "AC-003"

    def test_find_by_method_name_and_param(self, sample_requirements: list[SpecRequirement]):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        results = index.search("process_payment amount", top_k=1, expand_related=False)
        assert len(results) == 1
        assert results[0].req_id == "FR-002"


class SynonymMockEmbeddingClient(MockEmbeddingClient):
    """Mock-клиент, возвращающий вектор заданного канонического текста для синонимов."""

    def __init__(self, synonym_map: dict[str, str], dimension: int = 128):
        super().__init__(dimension=dimension)
        self.synonym_map = synonym_map

    def embed_text(self, text: str) -> list[float]:
        canonical = self.synonym_map.get(text, text)
        return super().embed_text(canonical)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class TestSemanticSearch:
    def test_find_by_paraphrased_meaning_without_lexical_overlap(self):
        req1 = SpecRequirement(
            req_id="FR-101",
            section="Безопасность",
            title="Авторизация",
            description="Проверка учетных данных пользователя и выдача разрешений.",
        )
        req2 = SpecRequirement(
            req_id="FR-102",
            section="Финансы",
            title="Счета",
            description="Калькуляция стоимости услуг и выставление счета.",
        )
        req3 = SpecRequirement(
            req_id="FR-103",
            section="Аналитика",
            title="Метрики",
            description="Построение диаграмм активности графиков.",
        )

        # "вход в личный кабинет" не имеет общих слов с требованиями, но семантически эквивалентен req1
        synonym_query = "вход в личный кабинет"
        synonym_map = {synonym_query: requirement_to_text(req1)}
        client = SynonymMockEmbeddingClient(synonym_map=synonym_map, dimension=128)

        index = HybridSpecIndex(embedding_client=client)
        index.build_index([req1, req2, req3])

        # BM25 для запроса дает 0 совпадений по всем требованиям
        # Семантический поиск по эмбеддингам находит именно FR-101
        results = index.search(synonym_query, top_k=1, expand_related=False)
        assert len(results) == 1
        assert results[0].req_id == "FR-101"
        assert results.is_fallback is False


class FixedEmbeddingClient(BaseEmbeddingClient):
    """Детерминированный клиент для явного тестирования RRF слияния."""

    def __init__(self, vector_map: dict[str, list[float]], default_dim: int = 4):
        self._vector_map = vector_map
        self._dim = default_dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        if text in self._vector_map:
            return l2_normalize(self._vector_map[text])
        return l2_normalize([0.0] * self._dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class TestRRFFusion:
    def test_rrf_scoring_and_ranking(self):
        """Проверка RRF-слияния (Reciprocal Rank Fusion k=60).

        Req B сбалансированно силен в обоих каналах (BM25 ранг 2, семантика ранг 1):
            RRF(B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.032522
        Req A побеждает в BM25 (ранг 1), но слаб в семантике (ранг 3):
            RRF(A) = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.032266
        Req B должен занять первое место благодаря синергии обоих каналов.
        """
        req_a = SpecRequirement(
            req_id="REQ-A",
            title="Токены аутентификации",
            description="auth_token auth_token auth_token многократное вхождение",
        )
        req_b = SpecRequirement(
            req_id="REQ-B",
            title="Сессия пользователя",
            description="auth_token единичное вхождение",
        )
        req_c = SpecRequirement(
            req_id="REQ-C",
            title="Уведомления",
            description="email рассылка без ключевых слов",
        )

        query = "auth_token"
        req_a_text = requirement_to_text(req_a)
        req_b_text = requirement_to_text(req_b)
        req_c_text = requirement_to_text(req_c)

        # Задаем векторы так, чтобы косинусное сходство с запросом было:
        # req_b > req_c > req_a
        vector_map = {
            query: [1.0, 0.0, 0.0, 0.0],
            req_b_text: [0.95, 0.05, 0.0, 0.0],  # rank 1 sem
            req_c_text: [0.70, 0.30, 0.0, 0.0],  # rank 2 sem
            req_a_text: [0.40, 0.60, 0.0, 0.0],  # rank 3 sem
        }

        client = FixedEmbeddingClient(vector_map=vector_map, default_dim=4)
        index = HybridSpecIndex(embedding_client=client, rrf_k=60)
        index.build_index([req_a, req_b, req_c])

        results = index.search(query, top_k=2, expand_related=False)
        assert len(results) == 2
        assert results[0].req_id == "REQ-B"
        assert results[1].req_id == "REQ-A"


class TestRelationExpansion:
    def test_expand_related_reqs_included(self, sample_requirements: list[SpecRequirement]):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        # FR-001 ссылается на EXC-001 и FR-002
        results_expanded = index.search("FR-001", top_k=1, expand_related=True)
        result_ids = [r.req_id for r in results_expanded]

        assert result_ids[0] == "FR-001"
        assert "EXC-001" in result_ids
        assert "FR-002" in result_ids
        assert len(result_ids) == 3

    def test_expand_related_disabled(self, sample_requirements: list[SpecRequirement]):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        results_no_expand = index.search("FR-001", top_k=1, expand_related=False)
        assert len(results_no_expand) == 1
        assert results_no_expand[0].req_id == "FR-001"

    def test_expand_related_no_infinite_loop_or_duplicates(self):
        # Взаимные циклические ссылки
        req1 = SpecRequirement(
            req_id="FR-1",
            description="Первое требование",
            related_req_ids=["FR-2", "FR-1"],
        )
        req2 = SpecRequirement(
            req_id="FR-2",
            description="Второе требование",
            related_req_ids=["FR-1", "NON-EXISTENT"],
        )

        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index([req1, req2])

        results = index.search("FR-1", top_k=1, expand_related=True)
        result_ids = [r.req_id for r in results]

        assert result_ids == ["FR-1", "FR-2"]
        assert len(result_ids) == len(set(result_ids))


class TestEdgeCases:
    def test_empty_index(self):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index([])

        results = index.search("какой-то запрос", top_k=5)
        assert results == []
        assert results.is_fallback is True
        assert index.is_fallback is True

    def test_empty_query_returns_safe_fallback(self, sample_requirements: list[SpecRequirement]):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        # Пустая строка
        res_empty = index.search("", top_k=2)
        assert len(res_empty) == 2
        assert res_empty.is_fallback is True
        assert index.is_fallback is True

        # Строка из одних пробелов
        res_spaces = index.search("   \n\t  ", top_k=3)
        assert len(res_spaces) == 3
        assert res_spaces.is_fallback is True

    def test_query_without_matches_returns_safe_fallback(self, sample_requirements: list[SpecRequirement]):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        # Запрос с несуществующими словами и высоким порогом семантики
        results = index.search(
            "абсолютно_неизвестный_запрос_xyz_999",
            top_k=2,
            min_semantic_score=0.9,
        )
        assert len(results) == 2
        assert results.is_fallback is True
        assert index.is_fallback is True

    def test_get_by_id(self, sample_requirements: list[SpecRequirement]):
        client = MockEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        # Точное совпадение
        req = index.get_by_id("FR-001")
        assert req is not None
        assert req.req_id == "FR-001"

        # Регистронезависимость и пробелы
        req_lower = index.get_by_id("  fr-001  ")
        assert req_lower is not None
        assert req_lower.req_id == "FR-001"

        # Несуществующий ID
        assert index.get_by_id("UNKNOWN-999") is None
        assert index.get_by_id("") is None


class TestBackwardCompatibility:
    def test_spec_index_initialization_and_search(self, sample_requirements: list[SpecRequirement]):
        # Старый интерфейс SpecIndex(requirements)
        spec_index = SpecIndex(sample_requirements)
        assert len(spec_index.requirements) == len(sample_requirements)

        results = spec_index.search("FR-001", top_k=1, expand_related=False)
        assert len(results) == 1
        assert results[0].req_id == "FR-001"

    def test_spec_index_set_requirements(self, sample_requirements: list[SpecRequirement]):
        spec_index = SpecIndex()
        assert len(spec_index.requirements) == 0

        spec_index.set_requirements(sample_requirements[:2])
        assert len(spec_index.requirements) == 2
        assert spec_index.get_by_id("FR-001") is not None
        assert spec_index.get_by_id("AC-003") is None


class ErrorEmbeddingClient(BaseEmbeddingClient):
    @property
    def dimension(self) -> int:
        return 384

    def embed_text(self, text: str) -> list[float]:
        raise EmbeddingConnectionError("Service unavailable")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


class TestResilience:
    def test_embedding_failure_falls_back_to_bm25(self, sample_requirements: list[SpecRequirement]):
        client = ErrorEmbeddingClient()
        index = HybridSpecIndex(embedding_client=client)
        index.build_index(sample_requirements)

        # Поиск не должен падать, если семантический API временно недоступен
        results = index.search("FR-001", top_k=1, expand_related=False)
        assert len(results) == 1
        assert results[0].req_id == "FR-001"
        assert results.is_fallback is False
