# Живая проверка адаптера GitHub MCP (Модуль 3, часть 3)

Проверка подтверждает на настоящем GitHub то, что до неё проверено только на имитации:

1. `get_commit("<sha>^N")` возвращает родителей коммита (GitHub MCP Server не отдаёт их сам);
2. список файлов merge-коммита GitHub строит относительно первого родителя (FR-003, AC-014);
3. список файлов PR строится от общей базы веток (merge-base);
4. постраничная выдача (больше 100 файлов), переименования, удаления, бинарные файлы, большой файл без patch.

Результат — протокол с ревизиями и SHA первого родителя (AC-014, TEST-007). Токена в протоколе нет.

## 0. Что нужно

- Сервер и переменные из [MCP_LIVE_SETUP.md](MCP_LIVE_SETUP.md); тест `real_server` проходит.
- git и аккаунт GitHub, на котором можно создать репозиторий.

Если у участника 5 уже есть тестовый репозиторий с эталонами, переходите к разделу 5.

## 1. Создать учебный репозиторий

Из папки SpecGuard:
```powershell
.\.venv\Scripts\python.exe -m specguard.mcp.make_test_repo ..\specguard-mcp-testrepo --repo ЛОГИН/specguard-mcp-testrepo
```
Появятся папка `..\specguard-mcp-testrepo` с историей коммитов и файл целей `..\specguard-mcp-testrepo-targets.json` рядом с ней (не внутри: ожидаемые SHA не публикуются).

## 2. Опубликовать на GitHub

1. Создайте на GitHub пустой репозиторий `specguard-mcp-testrepo` (Public или Private, без README, .gitignore и лицензии).
2. Выполните две команды `git -C ... remote add ...` и `git -C ... push ...`, которые напечатал генератор.
3. Откройте напечатанную ссылку `.../compare/main...feature/session-timeout` и нажмите «Create pull request». В новом репозитории PR получит номер #1. Если номер другой, исправьте `"target_ref": "PR#1"` в файле целей.

## 3. Токен

Fine-grained token: Repository access — Only select repositories → `specguard-mcp-testrepo`; Permissions — `Contents: Read-only`, `Pull requests: Read-only`.

```powershell
$env:GITHUB_PERSONAL_ACCESS_TOKEN = "github_pat_..."
$env:SPECGUARD_MCP_SERVER = "$HOME\tools\github-mcp-server\github-mcp-server.exe"
```

## 4. Запустить проверку

```powershell
.\.venv\Scripts\python.exe -m specguard.mcp.live_check --targets ..\specguard-mcp-testrepo-targets.json --clone ..\specguard-mcp-testrepo
```

Скрипт напечатает итог по каждой цели и общий вывод, а протокол сохранит в `%USERPROFILE%\specguard-live-protocols\mcp-live-ДАТА.md` и `.json`.

| Строка итога | Что значит «подтверждено» |
|---|---|
| Родители коммита через «<sha>^N» | адаптер определил родителей так же, как git |
| Файлы merge-коммита от первого родителя | список файлов GitHub совпал с `git diff <первый родитель> <merge>` |
| Файлы PR от merge-base | список файлов PR совпал с `git diff база...head` |
| Файлы обычных коммитов совпадают с git | файлы и статусы совпали, включая коммит на 151 файл |

Код завершения: `0` — всё подтверждено; `1` — есть расхождения или ошибки чтения (подробности в протоколе); `2` — не настроены сервер или токен.

Протокол (`.md`) отправьте в чат команды или Claude: по нему включается `merge_diff_semantics_verified` или выбирается обходной путь. Строка «несовпадение входа» означает, что репозиторий изменился после создания файла целей (TEST-007), а не ошибку адаптера.

## 5. С репозиторием участника 5

```powershell
.\.venv\Scripts\python.exe -m specguard.mcp.live_check --targets tests\ground_truth\benchmarks.json --clone ПУТЬ_К_КЛОНУ_ЕГО_РЕПОЗИТОРИЯ
```

Из файла эталонов читаются `target_repo` и `target_ref`; если указаны `expected_head_sha`, `expected_base_sha`, `expected_first_parent_sha`, `expected_parent_shas`, они сверяются. Ожидаемые выводы по требованиям ТЗ проверка не читает. Без `--clone` она только читает цели и записывает ревизии. Цели можно задать и вручную: `--repo owner/name --ref 7 --ref <sha>`.

## 6. Без GitHub

`--mock` выполняет ту же проверку на имитации, построенной по локальному клону. Так видно, как выглядит протокол, но поведение GitHub имитация не подтверждает (TEST-006).
