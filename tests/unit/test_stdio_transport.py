"""Tests for the live stdio transport to GitHub MCP Server (Module 3, part 2: FR-005, NFR-002, NFR-003).

Сервер здесь — учебная заглушка ``mcp_stub_server.py``: она говорит на том же
протоколе, что и настоящий GitHub MCP Server, и умеет имитировать отказы.
Проверку с настоящим сервером включает переменная SPECGUARD_TEST_MCP_SERVER
(путь к github-mcp-server); к GitHub она не обращается.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from specguard.mcp import (
    SOURCE_LIVE,
    GitHubMCPClient,
    GitSourceError,
    GitSourceErrorKind,
    MCPCompatibilityError,
    MCPConfigurationError,
    MCPTransportError,
    StdioMCPTransport,
    create_stdio_transport,
)
from specguard.mcp.github_client import MOCK_NOTE
from specguard.mcp.stdio_transport import (
    PINNED_SERVER_VERSION,
    SERVER_TOOLS,
    build_server_command,
    build_server_env,
    check_tool_schemas,
)

STUB = Path(__file__).with_name("mcp_stub_server.py")
TOKEN = "test-token-123"


def stub_command(*flags):
    return [sys.executable, str(STUB), *flags]


def stub_env():
    env = dict(os.environ)
    env["GITHUB_TOOLSETS"] = "all"  # должно быть вычищено из окружения сервера
    return build_server_env(TOKEN, env)


@pytest.fixture
def make_transport():
    created = []

    def factory(*flags, **kwargs):
        kwargs.setdefault("call_timeout", 10)
        kwargs.setdefault("startup_timeout", 20)
        transport = StdioMCPTransport(build_server_command(stub_command(*flags)), env=stub_env(), **kwargs)
        created.append(transport)
        return transport

    yield factory
    for transport in created:
        transport.close()


def has(lines, fragment):
    return any(fragment in line for line in lines)


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


COMMIT_ARGS = {"owner": "acme", "repo": "shop", "sha": "abc1234", "detail": "none"}


class TestHandshakeAndCalls:
    def test_handshake_records_server_protocol_and_tools(self, make_transport):
        transport = make_transport()
        transport.start()

        assert transport.server_info == {"name": "github-mcp-server", "version": PINNED_SERVER_VERSION}
        assert transport.protocol_version == "2025-06-18"
        assert sorted(transport.tools) == sorted(SERVER_TOOLS)
        assert transport.describe() == "Источник: github-mcp-server 2.0.3 по stdio, протокол MCP 2025-06-18."
        assert transport.diagnostics == []

    def test_adapter_reads_pr_and_merge_commit_over_stdio(self, make_transport):
        client = GitHubMCPClient(transport=make_transport(), retry_delays=())

        pr = client.get_pr_context("acme/shop", 7)
        merge = client.get_commit_context("acme/shop", "merge_abc")

        assert [f.filename for f in pr.files] == ["src/auth.py", "README.md"]
        assert pr.source == SOURCE_LIVE and MOCK_NOTE not in pr.limitations
        assert pr.diagnostics[0].startswith("Источник: github-mcp-server 2.0.3")
        assert merge.is_merge_commit is True
        assert merge.first_parent_sha == merge.base_sha == merge.parent_shas[0]

    def test_token_only_in_env_read_only_flags_and_no_foreign_github_vars(self, make_transport):
        result = make_transport().call_tool("debug_env", {})
        seen = json.loads(result.text)

        assert seen["token"] == TOKEN
        assert TOKEN not in " ".join(seen["argv"])
        assert "--read-only" in seen["argv"]
        assert "--tools=" + ",".join(SERVER_TOOLS) in seen["argv"]
        assert seen["read_only"] == "true"
        assert seen["leaked"] == []

    def test_unknown_tool_is_a_tool_error(self, make_transport):
        result = make_transport().call_tool("create_pull_request", {})

        assert result.is_error is True
        assert "unknown tool" in result.text

    def test_server_initiated_ping_is_answered(self, make_transport):
        result = make_transport("--ping-first").call_tool("get_commit", COMMIT_ARGS)

        assert result.is_error is False and json.loads(result.text)["sha"]

    def test_noise_on_stdout_is_skipped_and_noted(self, make_transport):
        transport = make_transport("--noise")

        result = transport.call_tool("get_commit", COMMIT_ARGS)

        assert result.is_error is False
        assert has(transport.diagnostics, "не сообщение MCP")

    def test_other_server_version_is_noted(self, make_transport):
        transport = make_transport("--server-version=2.1.0")
        transport.start()

        assert has(transport.diagnostics, "отличается от закреплённой 2.0.3")


class TestFailures:
    def test_hung_call_times_out_and_is_cancelled(self, make_transport):
        transport = make_transport("--hang-on=get_commit", call_timeout=0.5)
        client = GitHubMCPClient(transport=transport, retry_delays=())

        with pytest.raises(GitSourceError) as info:
            client.get_commit_context("acme/shop", "abc1234")

        assert info.value.kind is GitSourceErrorKind.TIMEOUT
        assert wait_until(lambda: has(transport.stderr_tail, "cancelled request"))
        # Сервер жив, другие инструменты отвечают.
        assert client.get_pr_context("acme/shop", 1).files

    def test_crash_restarts_once_then_gives_up(self, make_transport):
        transport = make_transport("--crash-after=1", max_restarts=1)

        assert transport.call_tool("get_commit", COMMIT_ARGS).is_error is False
        with pytest.raises(MCPTransportError) as crash:
            transport.call_tool("get_commit", COMMIT_ARGS)
        assert "simulated crash" in str(crash.value) and "код 2" in str(crash.value)

        assert transport.call_tool("get_commit", COMMIT_ARGS).is_error is False  # после перезапуска
        assert transport.restarts == 1 and has(transport.diagnostics, "перезапущен")
        with pytest.raises(MCPTransportError):
            transport.call_tool("get_commit", COMMIT_ARGS)
        with pytest.raises(MCPConfigurationError) as exhausted:
            transport.call_tool("get_commit", COMMIT_ARGS)
        assert "лимит перезапусков" in str(exhausted.value)

    def test_adapter_retry_survives_a_crash(self, make_transport):
        transport = make_transport("--crash-after=3", max_restarts=1)
        client = GitHubMCPClient(transport=transport, retry_delays=(0,))

        client.get_pr_context("acme/shop", 7)  # 3 вызова: get, get_files, get
        second = client.get_pr_context("acme/shop", 8)  # первый же вызов падает, повтор после перезапуска

        assert second.files and transport.restarts == 1
        assert has(second.diagnostics, "перезапущен")
        assert has(second.diagnostics, "повтор через 0 с")

    def test_server_exiting_at_start_is_configuration_error_with_its_message(self, make_transport):
        sleeps = []
        client = GitHubMCPClient(transport=make_transport("--crash-on-start"), sleep=sleeps.append)

        with pytest.raises(GitSourceError) as info:
            client.get_pr_context("acme/shop", 7)

        assert info.value.kind is GitSourceErrorKind.NOT_CONFIGURED
        assert "authentication required" in str(info.value)
        assert sleeps == []

    @pytest.mark.parametrize("flag, fragment", [
        ("--drop-tool=get_commit", "не предоставляет инструмент get_commit"),
        ("--drop-param=pull_request_read.pullNumber", "pull_request_read: нет параметров pullNumber"),
    ])
    def test_incompatible_server_is_reported_without_retries(self, make_transport, flag, fragment):
        transport = make_transport(flag)
        with pytest.raises(MCPCompatibilityError) as direct:
            transport.start()
        assert fragment in str(direct.value)

        sleeps = []
        client = GitHubMCPClient(transport=make_transport(flag), sleep=sleeps.append)
        with pytest.raises(GitSourceError) as info:
            client.get_pr_context("acme/shop", 7)
        assert info.value.kind is GitSourceErrorKind.INCOMPATIBLE and sleeps == []

    def test_slow_start_hits_startup_timeout(self, make_transport):
        with pytest.raises(MCPTransportError) as info:
            make_transport("--slow-start=5", startup_timeout=0.5).start()

        assert "не завершил запуск" in str(info.value)
        assert not isinstance(info.value, MCPConfigurationError)

    def test_missing_server_binary(self):
        transport = StdioMCPTransport([str(STUB.with_name("no-such-server")), "stdio"])

        with pytest.raises(MCPConfigurationError) as info:
            transport.start()

        assert "не найден исполняемый файл" in str(info.value)

    def test_close_stops_server_and_blocks_further_calls(self, make_transport):
        transport = make_transport()
        transport.start()
        process = transport._proc

        transport.close()

        assert process.poll() is not None
        with pytest.raises(MCPConfigurationError):
            transport.call_tool("get_commit", COMMIT_ARGS)


class TestConfiguration:
    def test_check_tool_schemas_accepts_matching_server(self):
        tools = {
            "pull_request_read": {"inputSchema": {
                "properties": {p: {} for p in ("method", "owner", "repo", "pullNumber", "page", "perPage")}
                | {"method": {"enum": ["get", "get_files", "get_diff"]}},
                "required": ["method", "owner", "repo", "pullNumber"]}},
            "get_commit": {"inputSchema": {
                "properties": {p: {} for p in ("owner", "repo", "sha", "page", "perPage")}
                | {"detail": {"enum": ["none", "stats", "full_patch"]}},
                "required": ["owner", "repo", "sha"]}},
        }
        assert check_tool_schemas(tools) == []

        tools["get_commit"]["inputSchema"]["properties"]["detail"]["enum"] = ["none", "stats"]
        tools["get_commit"]["inputSchema"]["required"].append("token")
        problems = check_tool_schemas(tools)
        assert any("full_patch" in p for p in problems) and any("token" in p for p in problems)

    def test_env_without_token_explains_what_to_set(self):
        with pytest.raises(MCPConfigurationError) as info:
            create_stdio_transport(environ={"PATH": ""})

        assert "GITHUB_PERSONAL_ACCESS_TOKEN" in str(info.value)

    def test_env_without_server_explains_what_to_set(self):
        with pytest.raises(MCPConfigurationError) as info:
            create_stdio_transport(environ={"GITHUB_PERSONAL_ACCESS_TOKEN": TOKEN, "PATH": ""})

        assert "SPECGUARD_MCP_SERVER" in str(info.value)

    def test_env_configuration_builds_read_only_command(self):
        environ = {"GITHUB_TOKEN": TOKEN, "SPECGUARD_MCP_SERVER": "/opt/github-mcp-server",
                   "SPECGUARD_MCP_TIMEOUT": "15", "GITHUB_TOOLSETS": "all", "PATH": ""}

        transport = create_stdio_transport(environ=environ)

        assert transport.command == [
            "/opt/github-mcp-server", "stdio", "--read-only", "--tools=" + ",".join(SERVER_TOOLS)]
        assert transport.call_timeout == 15.0
        assert transport._env["GITHUB_PERSONAL_ACCESS_TOKEN"] == TOKEN
        assert "GITHUB_TOOLSETS" not in transport._env and "GITHUB_TOKEN" not in transport._env

    def test_bad_timeout_value(self):
        with pytest.raises(MCPConfigurationError):
            create_stdio_transport("/opt/github-mcp-server", TOKEN,
                                   environ={"SPECGUARD_MCP_TIMEOUT": "soon", "PATH": ""})

    def test_client_starts_server_lazily_and_closes_it(self):
        with GitHubMCPClient(mcp_server_cmd=stub_command(), auth_token=TOKEN, retry_delays=()) as client:
            ctx = client.get_pr_context("acme/shop", 7)
            transport = client._transport
            assert ctx.files and transport._alive()
        assert not transport._alive()


@pytest.mark.skipif(not os.environ.get("SPECGUARD_TEST_MCP_SERVER"),
                    reason="задайте SPECGUARD_TEST_MCP_SERVER=путь к github-mcp-server")
def test_real_server_handshake_without_github_calls():
    # Токен не с префиксом ghp_: сервер не обращается к GitHub при запуске.
    transport = create_stdio_transport(os.environ["SPECGUARD_TEST_MCP_SERVER"], "handshake-only-token",
                                       environ=dict(os.environ))
    try:
        transport.start()
        assert transport.server_info["name"] == "github-mcp-server"
        assert sorted(transport.tools) == sorted(SERVER_TOOLS)
        assert not has(transport.diagnostics, "лишние инструменты")
        assert transport.call_tool("get_commit", {"owner": "x"}).is_error is True  # проверка аргументов на сервере
    finally:
        transport.close()
