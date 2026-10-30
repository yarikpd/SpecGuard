from abc import ABC, abstractmethod
import hashlib
import math
import random
from typing import Any

import httpx


class EmbeddingError(Exception):
    pass


class EmbeddingConnectionError(EmbeddingError):
    pass


class EmbeddingAPIError(EmbeddingError):
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def l2_normalize(vector: list[float]) -> list[float]:
    squared_sum = sum(x * x for x in vector)
    if squared_sum <= 0.0:
        return list(vector)
    norm = math.sqrt(squared_sum)
    if norm < 1e-12:
        return list(vector)
    return [x / norm for x in vector]


class BaseEmbeddingClient(ABC):
    _model_name: str = ""

    @property
    def model_name(self) -> str:
        return getattr(self, "_model_name", None) or self.__class__.__name__

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = str(value)

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        batch_size: int = 32,
        dimension: int | None = None,
        client: httpx.Client | None = None,
    ):
        self.endpoint = base_url
        self.base_url = base_url

        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self._dimension = dimension

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = client or httpx.Client(timeout=self.timeout, headers=headers)
        self._owns_client = client is None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed_text("probe")
            self._dimension = len(probe)
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        results = self.embed_batch([text])
        if not results:
            raise EmbeddingAPIError("Empty embedding result received from service")
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            embeddings = self._request_chunk(chunk)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def normalize_vector(self, vector: list[float]) -> list[float]:
        return l2_normalize(vector)

    def _request_chunk(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }

        try:
            response = self._client.post(self.endpoint, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise EmbeddingConnectionError(
                f"Failed to connect to embedding service at {self.endpoint}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingConnectionError(
                f"Request to embedding service timed out ({self.timeout}s): {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingAPIError(
                f"Embedding API HTTP {exc.response.status_code}: {exc.response.text}",
                status_code=exc.response.status_code,
                response_body=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingConnectionError(
                f"Network communication error with {self.endpoint}: {exc}"
            ) from exc

        try:
            body = response.json()
        except Exception as exc:
            raise EmbeddingAPIError(
                f"Invalid JSON received from embedding service: {response.text}"
            ) from exc

        raw_data = body.get("data")
        if not isinstance(raw_data, list):
            raise EmbeddingAPIError(f"Missing or malformed 'data' field in API response: {body}")

        ordered_items = sorted(raw_data, key=lambda item: item.get("index", 0))

        embeddings: list[list[float]] = []
        for item in ordered_items:
            vec = item.get("embedding")
            if not isinstance(vec, list):
                raise EmbeddingAPIError(f"Invalid embedding item in API response: {item}")
            norm_vec = self.normalize_vector(vec)
            embeddings.append(norm_vec)

        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        return embeddings

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenAIEmbeddingClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class MockEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, dimension: int = 384, model_name: str = "mock-embeddings"):
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        self._dimension = dimension
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self._generate_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(text) for text in texts]

    def _generate_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(seed)
        raw_vector = [rng.gauss(0.0, 1.0) for _ in range(self._dimension)]
        return l2_normalize(raw_vector)
