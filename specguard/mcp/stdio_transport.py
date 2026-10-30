"""Живой транспорт к GitHub MCP Server через stdio (Модуль 3, часть 2).

Запускает локальный процесс ``github-mcp-server stdio`` в режиме только
чтения и только с нужными инструментами и обменивается с ним сообщениями
JSON-RPC 2.0, по одному JSON-объекту на строку (MCP stdio).

Сверено с настоящим сервером v2.0.3:
- поддерживаются версии протокола 2024-11-05 … 2025-11-25;
- ``--read-only --tools=…`` оставляет ровно перечисленные инструменты;
- неверные аргументы приходят результатом с ``isError``, неизвестный
  инструмент — ошибкой JSON-RPC ``-32602``;
- при закрытии stdin сервер завершается с кодом 0, не дописывая ответы на
  незавершённые запросы, поэтому stdin держится открытым до ответа;
- журнал сервер пишет в stderr.

Токен передаётся только через переменную окружения процесса, не в
аргументах командной строки; переменные ``GITHUB_*`` родительского
окружения в процесс сервера не попадают.
"""

import atexit
import collections
import itertools
import json
import os
import shutil
import subprocess
import threading
import weakref
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Set

from specguard import __version__
from specguard.mcp.errors import redact_secrets
from specguard.mcp.transport import (
    SOURCE_LIVE,
    MCPCompatibilityError,
    MCPConfigurationError,
    MCPTimeoutError,
    MCPTransportError,
    ToolCallResult,
)

PROTOCOL_VERSION = "2025-06-18"
KNOWN_PROTOCOL_VERSIONS = frozenset({"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"})
PINNED_SERVER_VERSION = "2.0.3"
SERVER_TOOLS = ("pull_request_read", "get_commit", "get_file_contents")

# Что адаптер передаёт инструментам. Схема сервера обязана это принимать.
EXPECTED_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "pull_request_read": {
        "params": {"method", "owner", "repo", "pullNumber", "page", "perPage"},
        "enums": {"method": {"get", "get_files"}},
    },
    "get_commit": {
        "params": {"owner", "repo", "sha", "detail", "page", "perPage"},
        "enums": {"detail": {"none", "full_patch"}},
    },
}

STDERR_TAIL_LINES = 40
DEFAULT_CALL_TIMEOUT = 60.0
DEFAULT_STARTUP_TIMEOUT = 30.0


ENV_SERVER = "SPECGUARD_MCP_SERVER"
ENV_TIMEOUT = "SPECGUARD_MCP_TIMEOUT"
TOKEN_ENV_VARS = ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN")


def create_stdio_transport(
    server: Any = None,
    token: Optional[str] = None,
    *,
    call_timeout: Optional[float] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> "StdioMCPTransport":
    """Транспорт по настройкам: аргументы, иначе переменные окружения.

    Сервер: ``server`` (путь или команда списком), иначе ``SPECGUARD_MCP_SERVER``,
    иначе ``github-mcp-server`` из PATH. Токен: ``token``, иначе
    ``GITHUB_PERSONAL_ACCESS_TOKEN`` или ``GITHUB_TOKEN``. Таймаут вызова:
    ``call_timeout``, иначе ``SPECGUARD_MCP_TIMEOUT`` (секунды).
    """
    environ = os.environ if environ is None else environ
    token = token or next((environ[name] for name in TOKEN_ENV_VARS if environ.get(name)), None)
    if not token:
        raise MCPConfigurationError(
            "не задан токен GitHub: установите переменную GITHUB_PERSONAL_ACCESS_TOKEN "
            "(fine-grained токен только на чтение)"
        )
    if not server:
        server = environ.get(ENV_SERVER) or shutil.which("github-mcp-server", path=environ.get("PATH"))
    if not server:
        raise MCPConfigurationError(
            f"не найден github-mcp-server: укажите путь к нему в переменной {ENV_SERVER}"
        )
    if call_timeout is None:
        raw = environ.get(ENV_TIMEOUT)
        try:
            call_timeout = float(raw) if raw else DEFAULT_CALL_TIMEOUT
        except ValueError:
            raise MCPConfigurationError(f"{ENV_TIMEOUT} должна быть числом секунд, получено {raw!r}") from None
    return StdioMCPTransport(
        build_server_command(server), env=build_server_env(token, environ), call_timeout=call_timeout
    )


def build_server_command(server: Any, tools: Sequence[str] = SERVER_TOOLS) -> List[str]:
    """Команда запуска: путь к серверу (или команда списком) + stdio, только чтение, только нужные инструменты."""
    base = [server] if isinstance(server, (str, os.PathLike)) else list(server)
    return [os.fspath(part) for part in base] + ["stdio", "--read-only", "--tools=" + ",".join(tools)]


def build_server_env(token: str, base_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Окружение процесса сервера: без чужих GITHUB_*, с токеном и режимом только чтения."""
    source = os.environ if base_env is None else base_env
    env = {key: value for key, value in source.items() if not key.upper().startswith("GITHUB_")}
    env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token
    env["GITHUB_READ_ONLY"] = "true"
    return env


def check_tool_schemas(
    tools: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]] = EXPECTED_TOOL_SCHEMAS,
) -> List[str]:
    """Расхождения между схемами инструментов сервера и тем, что отправляет адаптер."""
    problems: List[str] = []
    for name, spec in expected.items():
        tool = tools.get(name)
        if tool is None:
            problems.append(f"сервер не предоставляет инструмент {name}")
            continue
        schema = tool.get("inputSchema") or {}
        props = schema.get("properties") or {}
        params: Set[str] = set(spec["params"])
        missing = sorted(params - set(props))
        if missing:
            problems.append(f"{name}: нет параметров {', '.join(missing)}")
        required_extra = sorted(set(schema.get("required") or []) - params)
        if required_extra:
            problems.append(f"{name}: сервер требует неизвестные адаптеру параметры {', '.join(required_extra)}")
        for param, values in spec.get("enums", {}).items():
            enum = (props.get(param) or {}).get("enum")
            if enum is not None and not set(values) <= set(enum):
                absent = ", ".join(sorted(set(values) - set(enum)))
                problems.append(f"{name}.{param}: сервер не принимает значения {absent}")
    return problems


class _Pending:
    __slots__ = ("event", "result", "error", "failure", "generation")

    def __init__(self, generation: int):
        self.event = threading.Event()
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None
        self.failure: Optional[MCPTransportError] = None
        self.generation = generation


class StdioMCPTransport:
    """Транспорт MCP поверх stdin/stdout дочернего процесса GitHub MCP Server.

    Сервер запускается при первом вызове (или ``start()``). Если процесс
    упал, следующий вызов перезапускает его не больше ``max_restarts`` раз.
    """

    source = SOURCE_LIVE

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Optional[Mapping[str, str]] = None,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        max_restarts: int = 1,
        expected_schemas: Mapping[str, Mapping[str, Any]] = EXPECTED_TOOL_SCHEMAS,
    ):
        if not command:
            raise ValueError("Пустая команда запуска сервера")
        self.command = [os.fspath(part) for part in command]
        self._env = dict(env) if env is not None else None
        self.call_timeout = call_timeout
        self.startup_timeout = startup_timeout
        self.max_restarts = max(0, max_restarts)
        self.expected_schemas = expected_schemas

        self.server_info: Dict[str, Any] = {}
        self.protocol_version: Optional[str] = None
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.diagnostics: List[str] = []
        self.restarts = 0

        self._proc: Optional[subprocess.Popen] = None
        self._generation = 0
        self._closed = False
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending: Dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._stderr_tail: Deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._ignored_lines = 0
        self._stderr_reader = threading.Thread()
        atexit.register(_close_quietly, weakref.ref(self))

    # ----------------------------------------------------------- публичное

    def start(self) -> None:
        """Запустить сервер и выполнить рукопожатие, если он ещё не работает."""
        with self._lock:
            if self._closed:
                raise MCPConfigurationError("транспорт закрыт")
            if self._alive():
                return
            if self._proc is not None:
                if self.restarts >= self.max_restarts:
                    raise MCPConfigurationError(
                        f"{self._describe_exit(self._proc)}; сервер падает повторно, лимит перезапусков исчерпан"
                    )
                self.restarts += 1
                self.diagnostics.append(
                    f"GitHub MCP Server перезапущен после сбоя ({self.restarts} из {self.max_restarts})."
                )
            self._spawn()
            self._handshake()

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolCallResult:
        self.start()
        try:
            result = self._request("tools/call", {"name": name, "arguments": dict(arguments)}, self.call_timeout)
        except _RemoteError as exc:
            # Ошибка протокола (например, неизвестный инструмент) — для адаптера это ошибка инструмента.
            return ToolCallResult(is_error=True, text=f"MCP error {exc.code}: {exc.message}")
        content = result.get("content") if isinstance(result, dict) else None
        texts = [
            item.get("text", "")
            for item in (content or [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return ToolCallResult(
            is_error=bool(result.get("isError")) if isinstance(result, dict) else True,
            text="\n".join(texts),
            structured=result.get("structuredContent") if isinstance(result, dict) else None,
        )

    def describe(self) -> str:
        """Происхождение данных для отчёта: какой сервер и протокол отвечали."""
        name = self.server_info.get("name", "github-mcp-server")
        version = self.server_info.get("version", "?")
        return f"Источник: {name} {version} по stdio, протокол MCP {self.protocol_version or '?'}."

    @property
    def stderr_tail(self) -> List[str]:
        return list(self._stderr_tail)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            proc, self._proc = self._proc, None
        if proc is not None:
            _stop_process(proc)
        self._fail_pending(MCPTransportError("транспорт закрыт"), generation=None)

    def __enter__(self) -> "StdioMCPTransport":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------- процесс

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                creationflags=creationflags,
            )
        except FileNotFoundError:
            raise MCPConfigurationError(f"не найден исполняемый файл сервера: {self.command[0]}") from None
        except OSError as exc:
            raise MCPConfigurationError(f"не удалось запустить сервер {self.command[0]}: {exc}") from None
        self._generation += 1
        self._proc = proc
        self._stderr_tail.clear()
        stderr_reader = threading.Thread(target=self._read_stderr, args=(proc,), daemon=True)
        stderr_reader.start()
        self._stderr_reader = stderr_reader
        threading.Thread(
            target=self._read_stdout, args=(proc, self._generation, stderr_reader), daemon=True
        ).start()

    def _handshake(self) -> None:
        params = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "specguard", "version": __version__},
        }
        try:
            init = self._request("initialize", params, self.startup_timeout)
            self._notify("notifications/initialized", {})
            tools = self._list_tools()
        except MCPTimeoutError:
            self._kill_current()
            raise MCPTransportError(f"сервер не завершил запуск за {self.startup_timeout:g} с") from None
        except _RemoteError as exc:
            self._kill_current()
            raise MCPCompatibilityError(f"сервер отклонил рукопожатие: {exc.message}") from None
        except MCPTransportError:
            # Процесс завершился при запуске: почти всегда настройка (токен, флаги).
            proc = self._proc
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self._stderr_reader.join(timeout=2)
            description = self._describe_exit(proc)
            self._kill_current()
            raise MCPConfigurationError(description) from None

        self.protocol_version = init.get("protocolVersion")
        info = init.get("serverInfo") or {}
        self.server_info = {"name": info.get("name"), "version": info.get("version")}
        if self.protocol_version not in KNOWN_PROTOCOL_VERSIONS:
            self.diagnostics.append(f"Сервер выбрал незнакомую версию протокола MCP: {self.protocol_version}.")
        if self.server_info.get("version") != PINNED_SERVER_VERSION:
            self.diagnostics.append(
                f"Версия сервера {self.server_info.get('version')} отличается от закреплённой {PINNED_SERVER_VERSION}; "
                "схемы инструментов проверены автоматически."
            )

        problems = check_tool_schemas(tools, self.expected_schemas)
        if problems:
            self._kill_current()
            raise MCPCompatibilityError("; ".join(problems))
        self.tools = tools
        extra = sorted(set(tools) - set(SERVER_TOOLS))
        if extra:
            self.diagnostics.append(f"Сервер открыл лишние инструменты (адаптер их не вызывает): {', '.join(extra)}.")
        not_read_only = sorted(
            name for name in set(tools) & set(self.expected_schemas)
            if (tools[name].get("annotations") or {}).get("readOnlyHint") is not True
        )
        if not_read_only:
            self.diagnostics.append(f"Инструменты без пометки «только чтение»: {', '.join(not_read_only)}.")

    def _list_tools(self) -> Dict[str, Dict[str, Any]]:
        tools: Dict[str, Dict[str, Any]] = {}
        cursor: Optional[str] = None
        for _ in range(50):
            result = self._request("tools/list", {"cursor": cursor} if cursor else {}, self.startup_timeout)
            for tool in result.get("tools") or []:
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    tools[tool["name"]] = tool
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def _kill_current(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            _stop_process(proc, graceful=False)

    # ------------------------------------------------------------ протокол

    def _request(self, method: str, params: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
        request_id = next(self._ids)
        pending = _Pending(self._generation)
        with self._pending_lock:
            self._pending[request_id] = pending
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)})
            if not pending.event.wait(timeout):
                self._notify_quietly(
                    "notifications/cancelled", {"requestId": request_id, "reason": "timeout"}
                )
                raise MCPTimeoutError(f"{method}: нет ответа за {timeout:g} с")
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
        if pending.failure is not None:
            raise pending.failure
        if pending.error is not None:
            raise _RemoteError(pending.error)
        return pending.result if isinstance(pending.result, dict) else {}

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def _notify_quietly(self, method: str, params: Mapping[str, Any]) -> None:
        try:
            self._notify(method, params)
        except MCPTransportError:
            pass

    def _send(self, message: Mapping[str, Any]) -> None:
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._write_lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise MCPTransportError(self._describe_exit(proc))
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except (OSError, ValueError):
                raise MCPTransportError(self._describe_exit(proc)) from None

    def _read_stdout(self, proc: subprocess.Popen, generation: int, stderr_reader: threading.Thread) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self._ignore(line)
                continue
            if not isinstance(message, dict):
                self._ignore(line)
            elif "method" in message and "id" in message:
                self._answer_server_request(message)
            elif "method" in message:
                continue  # уведомления сервера (журнал, изменения списков) адаптеру не нужны
            elif "id" in message:
                self._resolve(message)
            else:
                self._ignore(line)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        stderr_reader.join(timeout=2)
        self._fail_pending(MCPTransportError(self._describe_exit(proc)), generation=generation)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for raw in iter(proc.stderr.readline, b""):
            self._stderr_tail.append(redact_secrets(raw.decode("utf-8", errors="replace").rstrip()))

    def _resolve(self, message: Dict[str, Any]) -> None:
        with self._pending_lock:
            pending = self._pending.get(message.get("id"))  # type: ignore[arg-type]
        if pending is None:
            return  # ответ на отменённый по таймауту запрос
        if "error" in message:
            error = message["error"]
            pending.error = error if isinstance(error, dict) else {"message": str(error)}
        else:
            pending.result = message.get("result")
        pending.event.set()

    def _answer_server_request(self, message: Dict[str, Any]) -> None:
        if message.get("method") == "ping":
            reply: Dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        else:
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": f"Method not found: {message.get('method')}"},
            }
        try:
            self._send(reply)
        except MCPTransportError:
            pass

    def _ignore(self, line: str) -> None:
        self._ignored_lines += 1
        if self._ignored_lines <= 3:
            self.diagnostics.append(f"Сервер вывел в stdout не сообщение MCP, строка пропущена: {line[:80]!r}.")

    def _fail_pending(self, failure: MCPTransportError, generation: Optional[int]) -> None:
        with self._pending_lock:
            waiting = [p for p in self._pending.values() if generation is None or p.generation == generation]
        for pending in waiting:
            pending.failure = failure
            pending.event.set()

    def _describe_exit(self, proc: Optional[subprocess.Popen]) -> str:
        code = proc.poll() if proc is not None else None
        text = "процесс GitHub MCP Server завершился" + (f" (код {code})" if code is not None else "")
        tail = [line for line in self._stderr_tail if line.strip()][-5:]
        if tail:
            text += "; stderr: " + " | ".join(tail)
        return redact_secrets(text)


class _RemoteError(Exception):
    """Ответ JSON-RPC с ошибкой."""

    def __init__(self, error: Mapping[str, Any]):
        self.code = error.get("code")
        self.message = redact_secrets(str(error.get("message", "")))
        super().__init__(f"{self.code}: {self.message}")


def _stop_process(proc: subprocess.Popen, graceful: bool = True) -> None:
    """Корректно остановить сервер: закрыть stdin (сервер выходит сам), затем terminate/kill."""
    if proc.poll() is None:
        if graceful:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except (OSError, ValueError):
            pass


def _close_quietly(ref: "weakref.ReferenceType[StdioMCPTransport]") -> None:
    transport = ref()
    if transport is not None:
        try:
            transport.close()
        except Exception:  # noqa: BLE001 — при выходе интерпретатора ошибки неважны
            pass
