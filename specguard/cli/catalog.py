"""Read-only GitHub discovery for the human-operated repository picker."""
import json
import re

from specguard.cli.config import load_github_token, load_mcp_server
from specguard.mcp.github_client import parse_repo
from specguard.mcp.stdio_transport import create_stdio_transport

CATALOG_SCHEMAS = {
    "get_me": {"params": set()},
    "search_repositories": {"params": {"query", "page", "perPage"}},
    "list_pull_requests": {"params": {"owner", "repo", "state", "page", "perPage"}},
}


class GitHubCatalog:
    def __init__(self, transport=None):
        self.transport = transport or create_stdio_transport(
            server=load_mcp_server() or None, token=load_github_token(), tools=tuple(CATALOG_SCHEMAS), expected_schemas=CATALOG_SCHEMAS)

    def close(self):
        if hasattr(self.transport, "close"):
            self.transport.close()

    def _call(self, tool, arguments):
        if tool not in CATALOG_SCHEMAS:
            raise ValueError("Инструмент не разрешён")
        result = self.transport.call_tool(tool, arguments)
        if result.is_error:
            raise ValueError("GitHub MCP не смог прочитать список. Проверьте доступ токена.")
        return result.structured if result.structured is not None else json.loads(result.text)

    def repositories(self, page=1, query=""):
        if not query.strip():
            me = self._call("get_me", {})
            login = me.get("login")
            if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9-]+", login):
                raise ValueError("MCP не вернул имя GitHub-аккаунта")
            query = f"user:{login}"
        data = self._call("search_repositories", {"query": query, "page": page, "perPage": 50})
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("Неожиданный формат списка репозиториев MCP")
        return [(item["full_name"], item["full_name"]) for item in data["items"]]

    def pull_requests(self, repo, page=1):
        owner, name = parse_repo(repo)
        data = self._call("list_pull_requests", {"owner": owner, "repo": name,
                         "state": "open", "page": page, "perPage": 50})
        if not isinstance(data, list):
            raise ValueError("Неожиданный формат списка PR в MCP")
        return [(f"#{item['number']} · {item['title']}", str(item["number"])) for item in data]
