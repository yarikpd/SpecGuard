"""Граница между адаптером и MCP-сервером (Модуль 3).

Адаптер ``GitHubMCPClient`` не знает, как именно вызывается инструмент:
через локальный процесс GitHub MCP Server (stdio) или через имитацию
(``FakeMCPTransport``). Он видит только этот интерфейс. Поэтому логика
нормализации одна для живого и мок-режима, а отказы GitHub проверяются
«управляемыми ответами на границе адаптера» (TEST-006).
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

# Происхождение данных: живой GitHub MCP или имитация (AC-015).
SOURCE_LIVE = "github_mcp"
SOURCE_MOCK = "mock"


class MCPTransportError(Exception):
    """Вызов не выполнен: сервер недоступен, процесс завершился, соединение разорвано."""


class MCPTimeoutError(MCPTransportError):
    """Сервер не ответил за отведённое время."""


class MCPConfigurationError(MCPTransportError):
    """Сервер не запускается из-за настройки: нет файла, нет токена и т. п. Повтор не поможет."""


class MCPCompatibilityError(MCPTransportError):
    """Сервер работает, но его инструменты не совпадают с тем, что ждёт адаптер (NFR-003)."""


@dataclass(frozen=True)
class ToolCallResult:
    """Результат вызова MCP-инструмента (CallToolResult).

    ``text`` — склеенное текстовое содержимое ответа. GitHub MCP Server кладёт
    туда JSON (или текст ошибки при ``is_error``). ``structured`` —
    structuredContent, если сервер его прислал; адаптер на него не опирается,
    потому что его наличие зависит от версии протокола.
    """

    is_error: bool
    text: str
    structured: Optional[Any] = None


@runtime_checkable
class MCPTransport(Protocol):
    """Минимальный контракт транспорта: вызвать инструмент и вернуть результат.

    Ошибки уровня GitHub API приходят как ``ToolCallResult(is_error=True)``.
    Ошибки уровня транспорта — исключениями ``MCPTimeoutError``,
    ``MCPConfigurationError``, ``MCPCompatibilityError`` и
    ``MCPTransportError``.
    """

    source: str

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolCallResult:
        ...
