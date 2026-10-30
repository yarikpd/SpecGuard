from specguard.mcp.context import GitHubContext
from specguard.mcp.errors import GitSourceError, GitSourceErrorKind, classify_tool_error, redact_secrets
from specguard.mcp.fake_github import FakeFile, FakeGitHub, FakeMCPTransport, FakePullRequest, fake_sha
from specguard.mcp.github_client import (
    BaseGitClient,
    GitHubMCPClient,
    MockGitClient,
    MockGitHubClient,
    parse_repo,
)
from specguard.mcp.stdio_transport import StdioMCPTransport, create_stdio_transport
from specguard.mcp.transport import (
    SOURCE_LIVE,
    SOURCE_MOCK,
    MCPCompatibilityError,
    MCPConfigurationError,
    MCPTimeoutError,
    MCPTransport,
    MCPTransportError,
    ToolCallResult,
)

__all__ = [
    "BaseGitClient",
    "GitHubMCPClient",
    "MockGitClient",
    "MockGitHubClient",
    "parse_repo",
    "GitHubContext",
    "GitSourceError",
    "GitSourceErrorKind",
    "classify_tool_error",
    "redact_secrets",
    "FakeGitHub",
    "FakeFile",
    "FakePullRequest",
    "FakeMCPTransport",
    "fake_sha",
    "MCPTransport",
    "StdioMCPTransport",
    "create_stdio_transport",
    "MCPConfigurationError",
    "MCPCompatibilityError",
    "ToolCallResult",
    "MCPTransportError",
    "MCPTimeoutError",
    "SOURCE_LIVE",
    "SOURCE_MOCK",
]
