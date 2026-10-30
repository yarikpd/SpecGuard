"""Independent API configuration, persisted locally without API keys."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class APIConfig:
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)

    def validate(self):
        if not self.base_url.startswith(("http://", "https://")) or not self.model.strip():
            raise ValueError("Укажите API URL (http/https) и модель в настройках.")
        url = httpx.URL(self.base_url)
        if not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("API URL должен содержать адрес и путь без пароля, query или fragment.")
        if url.host == "generativelanguage.googleapis.com" and "/openai/" not in url.path:
            raise ValueError(
                "Указан нативный Gemini API, а SpecGuard использует OpenAI-совместимый формат. "
                "Для проверки задайте https://generativelanguage.googleapis.com/v1beta/openai/chat/completions; "
                "для embedding — https://generativelanguage.googleapis.com/v1beta/openai/embeddings. "
                "Адрес автоматически не изменяется."
            )

    def public(self):
        return {"base_url": self.base_url, "model": self.model}


@dataclass
class AppConfig:
    llm: APIConfig
    embedding: APIConfig


def config_path() -> Path:
    return Path(os.getenv("SPECGUARD_CONFIG", ".specguard/config.json"))


def load_config() -> AppConfig:
    path = config_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    secrets_file = path.parent / "secrets.json"
    secrets = json.loads(secrets_file.read_text(encoding="utf-8")) if secrets_file.exists() else {}
    def api(kind):
        saved = data.get(kind, {})
        prefix = f"SPECGUARD_{kind.upper()}_"
        return APIConfig(
            base_url=os.getenv(prefix + "URL", saved.get("base_url", "")),
            model=os.getenv(prefix + "MODEL", saved.get("model", "")),
            api_key=os.getenv(prefix + "API_KEY", secrets.get(kind, "")),
        )
    return AppConfig(llm=api("llm"), embedding=api("embedding"))


def save_config(config: AppConfig, kind: str | None = None):
    """Save one profile independently, preserving the other on-disk profile."""
    if kind not in (None, "llm", "embedding"):
        raise ValueError("Неизвестный профиль модели")
    path = config_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in ((kind,) if kind else ("llm", "embedding")):
        profile = getattr(config, name)
        if profile.base_url:
            APIConfig(profile.base_url, profile.model or "pending").validate()
        data[name] = profile.public()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_api_key(kind: str, key: str):
    """Store keys separately from settings with owner-only file permissions."""
    if kind not in ("llm", "embedding", "github"):
        raise ValueError("Неизвестный профиль ключа")
    path = config_path().parent / "secrets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if key:
        data[kind] = key
    else:
        data.pop(kind, None)
    # Open the destination directly with restrictive permissions; fchmod also
    # tightens permissions on a pre-existing file.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            json.dump(data, stream, ensure_ascii=False, indent=2)
    finally:
        if fd >= 0:
            os.close(fd)


def load_github_token() -> str:
    """Environment credentials take precedence over the locally saved token."""
    for name in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN"):
        if os.getenv(name):
            return os.environ[name]
    path = config_path().parent / "secrets.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return data.get("github", "")


def load_mcp_server() -> str:
    if os.getenv("SPECGUARD_MCP_SERVER"):
        return os.environ["SPECGUARD_MCP_SERVER"]
    path = config_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return data.get("github", {}).get("server", "")


def save_mcp_server(server: str):
    path = config_path()
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault("github", {})["server"] = server
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def discover_models(config: APIConfig) -> list[str]:
    # Models endpoints are optional in OpenAI-compatible servers.
    APIConfig(config.base_url, "probe").validate()
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        base = config.base_url.rstrip("/")
        for suffix in ("/chat/completions", "/embeddings"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        response = client.get(base + "/models", headers=headers)
        response.raise_for_status()
        data = response.json()
    models = sorted({item["id"] for item in data.get("data", [])
                     if isinstance(item, dict) and isinstance(item.get("id"), str)})
    if not models:
        raise ValueError("API не вернул список моделей. Введите модель вручную.")
    return models
