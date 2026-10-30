"""Расширенный контекст изменений адаптера GitHub MCP (Модуль 3).

Общий контракт ``GitContext`` в ``specguard/models/schema.py`` не меняется.
Адаптер возвращает ``GitHubContext`` — подкласс ``GitContext``: оркестратор
работает с ним как с обычным ``GitContext``, а дополнительные данные о
происхождении доступны тем, кому они нужны (CLI, отладка, будущий отчёт).
"""

from typing import List, Optional

from pydantic import Field

from specguard.models.schema import GitContext


class GitHubContext(GitContext):
    """GitContext с данными о происхождении, которые собирает адаптер."""

    # Родители коммита; None — неизвестны (GitHub MCP не возвращает родителей).
    parent_shas: Optional[List[str]] = None
    # Репозитории PR: head_repo отличается для fork и равен None, если fork
    # удалён или недоступен.
    base_repo: Optional[str] = None
    head_repo: Optional[str] = None
    # Head PR не изменился между чтениями; None — проверка не выполнялась.
    revisions_consistent: Optional[bool] = None
    # Происхождение данных: "github_mcp" (живой MCP) или "mock" (имитация).
    source: Optional[str] = None
    # Повторы и служебные наблюдения, не ограничивающие анализ.
    diagnostics: List[str] = Field(default_factory=list)
