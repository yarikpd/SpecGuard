"""SpecGuard Agent Package."""

from specguard.agent.llm_client import (
    BaseLLMClient,
    OpenAILLMClient,
    MockLLMClient,
    MockScenario,
    MockCall,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseParsingError,
    extract_json,
    extract_json_str,
)

try:
    from specguard.agent.orchestrator import SpecGuardOrchestrator
except ImportError:  # pragma: no cover
    SpecGuardOrchestrator = None  # type: ignore

__all__ = [
    "BaseLLMClient",
    "OpenAILLMClient",
    "MockLLMClient",
    "MockScenario",
    "MockCall",
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMAPIError",
    "LLMRateLimitError",
    "LLMResponseParsingError",
    "extract_json",
    "extract_json_str",
    "SpecGuardOrchestrator",
]
