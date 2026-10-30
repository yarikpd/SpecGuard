import math
from unittest.mock import MagicMock

import httpx
import pytest

from specguard.rag.embeddings import (
    BaseEmbeddingClient,
    EmbeddingAPIError,
    EmbeddingConnectionError,
    MockEmbeddingClient,
    OpenAIEmbeddingClient,
    l2_normalize,
)


def compute_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


class TestL2Normalize:
    def test_l2_normalize_basic(self):
        vec = [3.0, 4.0]
        norm_vec = l2_normalize(vec)
        assert norm_vec == pytest.approx([0.6, 0.8])
        assert compute_norm(norm_vec) == pytest.approx(1.0)

    def test_l2_normalize_zero_vector(self):
        vec = [0.0, 0.0, 0.0]
        assert l2_normalize(vec) == [0.0, 0.0, 0.0]


class TestMockEmbeddingClient:
    def test_default_dimension(self):
        client = MockEmbeddingClient()
        assert client.dimension == 384
        vec = client.embed_text("test requirement")
        assert len(vec) == 384

    def test_custom_dimension(self):
        client = MockEmbeddingClient(dimension=128)
        assert client.dimension == 128
        vec = client.embed_text("custom dimension test")
        assert len(vec) == 128

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="Dimension must be positive"):
            MockEmbeddingClient(dimension=0)
        with pytest.raises(ValueError, match="Dimension must be positive"):
            MockEmbeddingClient(dimension=-5)

    def test_determinism(self):
        client = MockEmbeddingClient(dimension=256)
        text = "FR-007: Independent embedding config"
        vec1 = client.embed_text(text)
        vec2 = client.embed_text(text)
        assert vec1 == vec2

    def test_different_texts_produce_different_vectors(self):
        client = MockEmbeddingClient(dimension=256)
        vec1 = client.embed_text("User authentication via JWT")
        vec2 = client.embed_text("Payment processing with Stripe")
        assert vec1 != vec2

    def test_unit_length_normalization(self):
        client = MockEmbeddingClient(dimension=384)
        for sample in ["Short", "A slightly longer text for checking normalization", "12345"]:
            vec = client.embed_text(sample)
            assert compute_norm(vec) == pytest.approx(1.0, abs=1e-5)

    def test_embed_batch(self):
        client = MockEmbeddingClient(dimension=64)
        texts = ["Text A", "Text B", "Text C"]
        batch_vecs = client.embed_batch(texts)
        assert len(batch_vecs) == 3
        for i, text in enumerate(texts):
            single_vec = client.embed_text(text)
            assert batch_vecs[i] == single_vec

    def test_embed_batch_empty(self):
        client = MockEmbeddingClient()
        assert client.embed_batch([]) == []


class TestOpenAIEmbeddingClient:
    def test_endpoint_construction(self):
        c1 = OpenAIEmbeddingClient(base_url="http://localhost:11434/v1", model_name="qwen3-emb")
        assert c1.endpoint == "http://localhost:11434/v1"
        c1.close()

        c2 = OpenAIEmbeddingClient(base_url="http://localhost:11434/v1/", model_name="qwen3-emb")
        assert c2.endpoint == "http://localhost:11434/v1/"
        c2.close()

        c3 = OpenAIEmbeddingClient(base_url="https://api.openai.com/v1/embeddings", model_name="text-embedding-3-small")
        assert c3.endpoint == "https://api.openai.com/v1/embeddings"
        c3.close()

    def test_auth_header(self):
        # Без API key
        c_no_key = OpenAIEmbeddingClient(base_url="http://localhost:11434/v1", model_name="qwen3-emb")
        assert "Authorization" not in c_no_key._client.headers
        c_no_key.close()

        # С API key
        c_with_key = OpenAIEmbeddingClient(
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small",
            api_key="sk-test-secret-token",
        )
        assert c_with_key._client.headers.get("Authorization") == "Bearer sk-test-secret-token"
        c_with_key.close()

    def test_embed_text_and_l2_normalization(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == "http://localhost:11434/v1"
            body = httpx.Response(200, json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [3.0, 4.0]}
                ],
                "model": "qwen3-embedding:0.6b",
            })
            return body

        mock_transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=mock_transport)

        emb_client = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="qwen3-embedding:0.6b",
            client=client,
        )

        vec = emb_client.embed_text("test string")
        assert vec == pytest.approx([0.6, 0.8])
        assert compute_norm(vec) == pytest.approx(1.0)
        assert emb_client.dimension == 2

    def test_embed_batch_chunking(self):
        sent_payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json
            data = json.loads(request.content.decode("utf-8"))
            sent_payloads.append(data)
            inputs = data["input"]
            resp_data = [
                {"object": "embedding", "index": idx, "embedding": [float(idx + 1), float(idx + 1)]}
                for idx in range(len(inputs))
            ]
            return httpx.Response(200, json={"data": resp_data})

        mock_transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=mock_transport)

        emb_client = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="qwen3-embedding",
            batch_size=2,
            client=client,
        )

        texts = ["t1", "t2", "t3", "t4", "t5"]
        results = emb_client.embed_batch(texts)

        assert len(sent_payloads) == 3
        assert sent_payloads[0]["input"] == ["t1", "t2"]
        assert sent_payloads[1]["input"] == ["t3", "t4"]
        assert sent_payloads[2]["input"] == ["t5"]
        assert len(results) == 5

    def test_respects_index_ordering(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ]
            })

        mock_transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=mock_transport)

        emb_client = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="bge-m3",
            client=client,
        )

        results = emb_client.embed_batch(["first", "second"])
        assert results[0] == pytest.approx([1.0, 0.0])
        assert results[1] == pytest.approx([0.0, 1.0])

    def test_http_500_error_handling(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error: model failed to load")

        mock_transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=mock_transport)

        emb_client = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="qwen3-embedding",
            client=client,
        )

        with pytest.raises(EmbeddingAPIError) as exc_info:
            emb_client.embed_text("failure test")

        assert exc_info.value.status_code == 500
        assert "500" in str(exc_info.value)
        assert "model failed to load" in exc_info.value.response_body

    def test_connect_error_handling(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused to 127.0.0.1:11434")

        mock_transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=mock_transport)

        emb_client = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="qwen3-embedding",
            client=client,
        )

        with pytest.raises(EmbeddingConnectionError) as exc_info:
            emb_client.embed_text("offline test")

        assert "Failed to connect" in str(exc_info.value)

    def test_timeout_error_handling(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Read timeout exceeded")

        mock_transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=mock_transport)

        emb_client = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="qwen3-embedding",
            client=client,
        )

        with pytest.raises(EmbeddingConnectionError) as exc_info:
            emb_client.embed_text("timeout test")

        assert "timed out" in str(exc_info.value)

    def test_dimension_predefined_vs_probe(self):
        # Predefined dimension doesn't call API
        c_predefined = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="test",
            dimension=768,
        )
        assert c_predefined.dimension == 768
        c_predefined.close()

        # Dynamic probe
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 2.0, 3.0]}]
            })

        mock_transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=mock_transport)
        c_probe = OpenAIEmbeddingClient(
            base_url="http://localhost:11434/v1",
            model_name="test",
            client=client,
        )
        assert c_probe.dimension == 3

    def test_context_manager(self):
        with OpenAIEmbeddingClient(base_url="http://localhost:11434/v1", model_name="test") as client:
            assert isinstance(client, BaseEmbeddingClient)
            assert not client._client.is_closed
        assert client._client.is_closed
