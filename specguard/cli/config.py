"""Configuration for SpecGuard API clients."""

import os
from dataclasses import dataclass


@dataclass
class APIConfig:
    base_url: str = ""
    model: str = ""
    api_key: str = ""


@dataclass
class AppConfig:
    llm: APIConfig
    embedding: APIConfig


def load_config() -> AppConfig:
    """Load independent LLM and embedding settings from environment."""

    return AppConfig(
        llm=APIConfig(
            base_url=os.getenv("SPECGUARD_LLM_URL", ""),
            model=os.getenv("SPECGUARD_LLM_MODEL", ""),
            api_key=os.getenv("SPECGUARD_LLM_API_KEY", ""),
        ),
        embedding=APIConfig(
            base_url=os.getenv("SPECGUARD_EMBEDDING_URL", ""),
            model=os.getenv("SPECGUARD_EMBEDDING_MODEL", ""),
            api_key=os.getenv("SPECGUARD_EMBEDDING_API_KEY", ""),
        ),
    )
