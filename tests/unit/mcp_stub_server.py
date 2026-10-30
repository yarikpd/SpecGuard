"""Учебная заглушка GitHub MCP Server для тестов stdio-транспорта (Модуль 3).

Говорит на том же протоколе, что и настоящий сервер v2.0.3 (JSON-RPC по
строкам через stdin/stdout), а данные берёт из имитации ``FakeGitHub``.
Флаги перед ``stdio`` включают отказы: зависание, падение, мусор в stdout,
ping от сервера, несовместимую схему, медленный старт.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from specguard.mcp.fake_github import FakeFile, FakeGitHub  # noqa: E402

PAGINATION = {"page": {"type": "number"}, "perPage": {"type": "number"}}
TOOLS = {
    "pull_request_read": {
        "properties": {
            "method": {"type": "string", "enum": ["get", "get_diff", "get_files", "get_commits"]},
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "pullNumber": {"type": "number"},
            **PAGINATION,
        },
        "required": ["method", "owner", "repo", "pullNumber"],
    },
    "get_commit": {
        "properties": {
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "sha": {"type": "string"},
            "detail": {"type": "string", "enum": ["none", "stats", "full_patch"]},
            **PAGINATION,
        },
        "required": ["owner", "repo", "sha"],
    },
    "get_file_contents": {
        "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}, "path": {"type": "string"}},
        "required": ["owner", "repo"],
    },
}


def send(message):
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def reply(request_id, result):
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def fail(request_id, code, message):
    send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def tool_list(args):
    enabled = [name.strip() for name in (args.tools or "").split(",") if name.strip()] or list(TOOLS)
    tools = []
    for name in enabled:
        if name not in TOOLS or name == args.drop_tool:
            continue
        schema = json.loads(json.dumps(TOOLS[name]))
        if args.drop_param and args.drop_param.startswith(name + "."):
            param = args.drop_param.split(".", 1)[1]
            schema["properties"].pop(param, None)
            schema["required"] = [p for p in schema["required"] if p != param]
        tools.append({
            "name": name,
            "inputSchema": {"type": "object", **schema},
            "annotations": {"readOnlyHint": True},
        })
    return tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--tools")
    parser.add_argument("--hang-on")
    parser.add_argument("--crash-after", type=int)
    parser.add_argument("--crash-on-start", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--ping-first", action="store_true")
    parser.add_argument("--drop-tool")
    parser.add_argument("--drop-param")
    parser.add_argument("--slow-start", type=float, default=0.0)
    parser.add_argument("--server-version", default="2.0.3")
    args = parser.parse_args()

    if args.crash_on_start or not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        sys.stderr.write("authentication required: set GITHUB_PERSONAL_ACCESS_TOKEN\n")
        sys.stderr.flush()
        sys.exit(1)
    if args.slow_start:
        time.sleep(args.slow_start)
    sys.stderr.write("GitHub MCP Server running on stdio\n")
    sys.stderr.flush()

    github = FakeGitHub(auto_provision=True, default_files=[
        FakeFile("src/auth.py", patch="@@ -1 +1,2 @@\n+MIN_PASSWORD_LENGTH = 8"),
        FakeFile("README.md", status="added", patch="@@ -0,0 +1 @@\n+# Shop"),
    ])
    names = {tool["name"] for tool in tool_list(args)}
    calls = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method, request_id = message.get("method"), message.get("id")
        if method is None:
            continue
        if request_id is None:
            if method == "notifications/cancelled":
                sys.stderr.write(f"cancelled request {message.get('params', {}).get('requestId')}\n")
                sys.stderr.flush()
            continue
        if method == "initialize":
            reply(request_id, {
                "protocolVersion": message["params"].get("protocolVersion"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "github-mcp-server", "version": args.server_version},
            })
        elif method == "tools/list":
            reply(request_id, {"tools": tool_list(args)})
        elif method == "ping":
            reply(request_id, {})
        elif method == "tools/call":
            calls += 1
            if args.crash_after is not None and calls > args.crash_after:
                sys.stderr.write("panic: simulated crash\n")
                sys.stderr.flush()
                sys.exit(2)
            name = message["params"]["name"]
            arguments = message["params"].get("arguments") or {}
            if name == args.hang_on:
                continue
            if args.ping_first:
                send({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"})
                answer = json.loads(sys.stdin.readline())
                if answer.get("id") != "server-ping" or "result" not in answer:
                    sys.stderr.write("client did not answer ping\n")
                    sys.exit(3)
            if args.noise:
                sys.stdout.write("this line is not JSON\n")
                send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "x"}})
            if name == "debug_env":
                leaked = sorted(k for k in os.environ if k.upper().startswith("GITHUB_")
                                and k.upper() not in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_READ_ONLY"))
                payload = {"argv": sys.argv[1:], "token": os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"),
                           "read_only": os.environ.get("GITHUB_READ_ONLY"), "leaked": leaked}
                reply(request_id, {"content": [{"type": "text", "text": json.dumps(payload)}]})
            elif name not in names:
                fail(request_id, -32602, f'unknown tool "{name}"')
            else:
                result = github.handle(name, arguments)
                reply(request_id, {"content": [{"type": "text", "text": result.text}], "isError": result.is_error})
        else:
            fail(request_id, -32601, f"method not found: {method}")


if __name__ == "__main__":
    main()
