# Живой режим GitHub MCP (Модуль 3)

Адаптер `GitHubMCPClient` сам запускает локальный GitHub MCP Server по stdio, когда нужен первый запрос. Сервер стартует в режиме только чтения и только с инструментами `pull_request_read`, `get_commit`, `get_file_contents`.

## 1. Сервер

Нужна закреплённая версия **v2.0.3** со страницы релизов: https://github.com/github/github-mcp-server/releases/tag/v2.0.3

| Платформа | Файл | SHA-256 |
|---|---|---|
| Windows x64 | `github-mcp-server_Windows_x86_64.zip` | `f0e1b9880a1b88730d1250d1c25cfee39617ec3e800a7a252f2279cd0c8394e0` |
| Windows ARM | `github-mcp-server_Windows_arm64.zip` | `d1321c411023c9a08e7709c2e9689551e3bff5a57033a3e5ff5f2c1db0c67a6f` |
| macOS Apple Silicon | `github-mcp-server_Darwin_arm64.tar.gz` | `5c95c01f75a7ec2370607e989b2968d314b02a54d78af8a9fdf38f3dfe6135f4` |
| macOS Intel | `github-mcp-server_Darwin_x86_64.tar.gz` | `115350858c3287843274efbf45e821d98f15d3304cb2c074740461ed32b45598` |
| Linux x64 | `github-mcp-server_Linux_x86_64.tar.gz` | `f797c20fe02a755865d0e2d46184ab47e31bf1619b0855b8d89c531e45a651c2` |

Проверка скачанного файла: `Get-FileHash .\github-mcp-server_Windows_x86_64.zip` (PowerShell) или `shasum -a 256 <файл>` (macOS/Linux). Сумма должна совпасть с таблицей. Затем распакуйте архив в любую папку вне репозитория.

## 2. Токен

Fine-grained personal access token (GitHub → Settings → Developer settings):
- Repository access — только нужные репозитории;
- Permissions — `Contents: Read-only`, `Pull requests: Read-only` (`Metadata: Read-only` добавится сам).

Токен не коммитится, не пересылается и не попадает в аргументы процесса, журналы и отчёты: адаптер передаёт его серверу только через окружение.

## 3. Переменные окружения

| Переменная | Назначение |
|---|---|
| `GITHUB_PERSONAL_ACCESS_TOKEN` | токен (если нет — берётся `GITHUB_TOKEN`) |
| `SPECGUARD_MCP_SERVER` | путь к `github-mcp-server` (если нет — ищется в PATH) |
| `SPECGUARD_MCP_TIMEOUT` | таймаут одного запроса, секунды (по умолчанию 60) |

PowerShell, только для текущего окна терминала:
```powershell
$env:GITHUB_PERSONAL_ACCESS_TOKEN = "github_pat_..."
$env:SPECGUARD_MCP_SERVER = "C:\tools\github-mcp-server\github-mcp-server.exe"
```

## 4. Использование

```python
from specguard.mcp import GitHubMCPClient

with GitHubMCPClient() as client:           # сервер остановится при выходе из with
    ctx = client.get_context("owner/repo", "42")       # PR #42
    ctx = client.get_context("owner/repo", "a1b2c3d")  # коммит
```

CLI в режиме `--live` создаёт `GitHubMCPClient()` и использует те же переменные.

При запуске адаптер проверяет: версию протокола MCP, версию сервера (отличие от 2.0.3 попадает в `diagnostics`), состав инструментов и их параметры. Ошибки настройки дают `GitSourceError` вида `not_configured`, несовпадение схем — `incompatible`; они не повторяются. Таймауты и падения сервера повторяются ограниченно, упавший сервер перезапускается один раз.

## 5. Проверка без обращения к GitHub

```powershell
$env:SPECGUARD_TEST_MCP_SERVER = "C:\tools\github-mcp-server\github-mcp-server.exe"
python -m pytest tests/unit/test_stdio_transport.py -k real_server -v
```

Тест запускает настоящий сервер с фиктивным токеном, проверяет рукопожатие и инструменты и не делает запросов к GitHub.
