"""Типизированные ошибки источника изменений (Модуль 3).

Адаптер бросает ``GitSourceError`` только тогда, когда без ответа анализ
невозможен (нет метаданных PR, коммит не найден). Частичные сбои — страница
файлов, проверка SHA, определение родителей — не прерывают чтение, а
попадают в ``GitContext.limitations`` (FR-009, FR-013).
"""

import re
from enum import Enum
from typing import Optional


class GitSourceErrorKind(str, Enum):
    NOT_FOUND_OR_NO_ACCESS = "not_found_or_no_access"
    INVALID_REVISION = "invalid_revision"
    AUTH_FAILED = "auth_failed"
    FORBIDDEN = "forbidden"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    PROTOCOL = "protocol"
    INCOMPATIBLE = "incompatible"
    NOT_CONFIGURED = "not_configured"
    TOOL_ERROR = "tool_error"


# Повторяются только сбои, которые могут пройти сами. Лимит GitHub не
# повторяется: ждать нужно минуты, а пользователь согласовал частичный отчёт.
RETRYABLE_KINDS = frozenset({GitSourceErrorKind.TIMEOUT, GitSourceErrorKind.UNAVAILABLE})

_KIND_TEXT = {
    GitSourceErrorKind.NOT_FOUND_OR_NO_ACCESS: (
        "не найдено или нет доступа (GitHub отвечает 404 и на отсутствующий, и на недоступный ресурс)"
    ),
    GitSourceErrorKind.INVALID_REVISION: "ревизия не найдена",
    GitSourceErrorKind.AUTH_FAILED: "ошибка аутентификации, проверьте токен",
    GitSourceErrorKind.FORBIDDEN: "доступ запрещён",
    GitSourceErrorKind.RATE_LIMITED: "превышен лимит запросов GitHub",
    GitSourceErrorKind.TIMEOUT: "превышено время ожидания ответа",
    GitSourceErrorKind.UNAVAILABLE: "источник временно недоступен",
    GitSourceErrorKind.PROTOCOL: "некорректный ответ MCP-сервера",
    GitSourceErrorKind.INCOMPATIBLE: "GitHub MCP Server несовместим с адаптером",
    GitSourceErrorKind.NOT_CONFIGURED: "живое подключение к GitHub MCP не настроено",
    GitSourceErrorKind.TOOL_ERROR: "ошибка инструмента MCP",
}

MAX_DETAIL_CHARS = 300

_SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9_.\-]{20,}"),
)


def redact_secrets(text: str) -> str:
    """Скрывает токены GitHub: секреты не должны попадать в сохраняемые материалы."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[скрыто]", text)
    return text


class GitSourceError(RuntimeError):
    """Не удалось получить данные, без которых анализ невозможен."""

    def __init__(
        self,
        kind: GitSourceErrorKind,
        detail: str = "",
        *,
        operation: Optional[str] = None,
        attempts: int = 1,
    ):
        self.kind = GitSourceErrorKind(kind)
        detail = redact_secrets(" ".join(str(detail).split()))
        if len(detail) > MAX_DETAIL_CHARS:
            detail = detail[:MAX_DETAIL_CHARS] + "…"
        self.detail = detail
        self.operation = operation
        self.attempts = attempts
        super().__init__(self.kind.value)

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    @property
    def summary(self) -> str:
        return _KIND_TEXT[self.kind]

    def __str__(self) -> str:
        text = self.summary
        if self.operation:
            text = f"{self.operation}: {text}"
        if self.detail:
            text += f" [{self.detail}]"
        if self.attempts > 1:
            text += f" (попыток: {self.attempts})"
        return text


# Формат go-github: "GET https://api.github.com/...: 404 Not Found []",
# а для синтетических ошибок сервера: "unexpected status 502: ...".
# После двоеточия обязателен пробел, чтобы порт в URL не принять за статус.
_HTTP_STATUS_RE = re.compile(r"(?:https?://\S+?:\s+|unexpected status\s+)(\d{3})\b")


def classify_tool_error(text: str, operation: Optional[str] = None) -> GitSourceError:
    """Сопоставляет текст ошибки GitHub MCP Server с видом ошибки.

    Тексты взяты из github-mcp-server v2.0.3 (pkg/errors/error.go). Ответ 422
    сервер передаёт без кода и URL, только сообщением GitHub, поэтому
    «No commit found» распознаётся по тексту.
    """
    lowered = text.lower()
    match = _HTTP_STATUS_RE.search(text)
    status = int(match.group(1)) if match else None

    if "rate limit exceeded" in lowered:
        kind = GitSourceErrorKind.RATE_LIMITED
    elif "no commit found" in lowered:
        kind = GitSourceErrorKind.INVALID_REVISION
    elif status == 404:
        kind = GitSourceErrorKind.NOT_FOUND_OR_NO_ACCESS
    elif status == 401:
        kind = GitSourceErrorKind.AUTH_FAILED
    elif status == 403:
        kind = GitSourceErrorKind.FORBIDDEN
    elif status is not None and status >= 500:
        kind = GitSourceErrorKind.UNAVAILABLE
    elif status is None and ("timeout" in lowered or "deadline exceeded" in lowered):
        kind = GitSourceErrorKind.TIMEOUT
    else:
        kind = GitSourceErrorKind.TOOL_ERROR

    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return GitSourceError(kind, first_line, operation=operation)
