# Project: SpecGuard - Participant 1 (Agent Core & Orchestration)

## Architecture
SpecGuard is an automated system for auditing Pull Requests and Git commits against technical specifications (ТЗ).
Participant 1 is responsible for the Agent Core, Orchestration, LLM client integration, Prompt Engineering, Prompt Injection Defense, Schema definition, and the Dispute procedure.

### Component Boundaries & Data Flow
```
User / CLI / Test Runner
          │
          ▼
┌────────────────────────────────────────────────────────┐
│               SpecGuardOrchestrator                    │
│             (specguard/agent/orchestrator.py)          │
└────────────────────────────────────────────────────────┘
     │                           │                   │
     ▼                           ▼                   ▼
┌──────────────┐       ┌───────────────────┐    ┌────────────────────┐
│ BaseGitClient│       │    SpecIndex      │    │   BaseLLMClient    │
│(Mock/GitHub) │       │(specguard/rag)    │    │ (OpenAI/Mock/Local)│
└──────────────┘       └───────────────────┘    └────────────────────┘
     │                           │                   │
     └───────────────────────────┴───────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │      AuditReport      │
                     │(Canonical Pydantic v2)│
                     └───────────────────────┘
                                 │
                                 ▼ (Dispute Workflow)
                     ┌───────────────────────┐
                     │    DisputeResponse    │
                     └───────────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Packaging & Schema Contracts | Add setuptools package discovery to pyproject.toml and complete Pydantic schemas (DisputeResponse, validators) in schema.py | M1 | Survey, Codebase Explorer |
| 2 | LLM Client Interface & OpenAILLMClient | Configurable OpenAI-compatible client supporting custom URL, API key, model, timeouts, JSON parsing | M2 | R2, FR-007, AC-004 |
| 3 | Offline MockLLMClient | Deterministic, fixture-driven mock client executing all audit and dispute scenarios offline without API keys | M2 | R5, TEST-006 |
| 4 | Prompt Injection Defense | XML sandboxing and raw data isolation preventing diffs/specs/summaries from overriding system instructions | M3 | R3, NFR-002, AC-009 |
| 5 | System Prompts Engineering | Structured prompts for Violation Detection (CONFIRMED vs SUSPECTED) and Implementation Tracking (IMPLEMENTED vs NOT_AFFECTED) with language support | M3 | R2, FR-004, FR-008 |
| 6 | Core Audit Orchestration | Orchestrate GitContext, RAG search, LLM analysis, and build structured AuditReport with quotes and lines | M4 | R1, FR-003, AC-002 |
| 7 | Merge Commit First-Parent Tracking | Detect merge commits, diff against first parent SHA, record first_parent_sha as base_parent_sha | M4 | R1, TEST-002, AC-014 |
| 8 | Partial Data & Truncation Handling | Handle truncated diffs, inaccessible bases, skipped files with is_partial=True and populated limitations/missing_data | M4 | R3, FR-009, FR-013 |
| 9 | Dispute Review Procedure | Hermetic re-evaluation of contested findings on saved audit artifacts without live network calls | M4 | R4, FR-012, AC-008 |
| 10 | E2E Opaque-Box Test Suite | Comprehensive 4-tier requirement-driven test suite validating all user requirements offline | M-E2E | R5, AC-001..AC-016 |
| 11 | E2E 100% Pass & Adversarial Hardening | Verify 100% pass of E2E suite (Tiers 1-4) and harden with Tier 5 white-box adversarial stress tests | M5 | Project Pattern DoD |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M-E2E | E2E Testing Suite | Opaque-box test suite across Tiers 1-4, test runner, TEST_INFRA.md, publish TEST_READY.md | none | DONE (108/108 E2E tests pass, TEST_READY.md published) |
| M1 | Packaging & Schema Contracts | Fix pyproject.toml package discovery, update specguard/models/schema.py | none | DONE (140 tests pass) |
| M2 | LLM Client & Mock Engine | Implement BaseLLMClient, OpenAILLMClient, MockLLMClient in specguard/agent/llm_client.py | M1 | DONE (Gate PASSED, 56 unit + 80 stress tests pass) |
| M3 | Prompts & Injection Defense | Implement specguard/agent/prompts.py and specguard/agent/defense.py | M1 | DONE (Gate PASSED, 82 unit/stress tests pass, 358 full project tests pass) |
| M4 | Agent Core & Dispute | Implement SpecGuardOrchestrator in specguard/agent/orchestrator.py with dispute review | M2, M3 | DONE (Gate PASSED, 529 project tests pass, 97% module coverage) |
| M5 | E2E Pass & Adversarial Hardening | Pass 100% E2E tests (Phase 1) + Adversarial coverage hardening (Phase 2 Tier 5) | M4, M-E2E | DONE (Gate PASSED, 145/145 E2E tests pass, 674 project tests pass, 100% statement coverage) |

## Interface Contracts

### 1. `specguard.models.schema`
- `RequirementStatus`: Enum (`IMPLEMENTED`, `VIOLATED`, `NOT_AFFECTED`)
- `ViolationSeverity`: Enum (`CONFIRMED`, `SUSPECTED`)
- `Evidence`: `file_path: str`, `revision: str`, `line_start: Optional[int]`, `line_end: Optional[int]`, `code_quote: str`, `spec_quote: str`
- `Finding`: `req_id: str`, `title: str`, `status: RequirementStatus`, `severity: Optional[ViolationSeverity]`, `description: str`, `evidence: List[Evidence]`, `counter_evidence: List[Evidence]`, `limitations: List[str]`
- `GitContext`: `target_repo: str`, `target_ref: str`, `base_sha: Optional[str]`, `head_sha: str`, `is_merge_commit: bool`, `first_parent_sha: Optional[str]`, `files: List[FileChange]`, `limitations: List[str]`, `is_partial: bool`
- `AuditReport`: `audit_id: str`, `timestamp: str`, `target_repo: str`, `target_ref: str`, `base_parent_sha: Optional[str]`, `user_summary: str`, `language: str`, `findings: List[Finding]`, `limitations: List[str]`, `missing_data: List[str]`, `is_partial: bool`, `metadata: Dict[str, Any]`
- `DisputeRequest`: `audit_id: str`, `finding_req_id: str`, `user_comment: str`
- `DisputeResponse`: `audit_id: str`, `finding_req_id: str`, `original_status: RequirementStatus`, `original_severity: Optional[ViolationSeverity]`, `dispute_status: str`, `updated_finding: Finding`, `reasoning: str`, `timestamp: str`

### 2. `specguard.agent.llm_client` ↔ `specguard.agent.orchestrator`
```python
class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[str] = "json",
    ) -> str: ...
```

### 3. `specguard.agent.defense` ↔ `specguard.agent.prompts`
- `isolate_raw_data(tag: str, content: str) -> str`: wraps content in `<raw_{tag}>` with closing tag sanitization / escaping.
- System prompt instruction ensures untrusted data inside raw tags is never executed.

### 4. `specguard.agent.orchestrator`
- `run_audit(repo: str, target_ref: str, user_summary: str = "", git_context: Optional[GitContext] = None, language: str = "ru") -> AuditReport`
- `review_dispute(dispute: DisputeRequest, saved_report: AuditReport, spec_index: Optional[SpecIndex] = None) -> DisputeResponse`

## Code Layout
```
specguard/
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py    # Core audit orchestrator & dispute review (M4)
│   ├── llm_client.py      # BaseLLMClient, OpenAILLMClient, MockLLMClient (M2)
│   ├── prompts.py         # System prompt templates (M3)
│   └── defense.py         # Prompt injection defense & XML data isolation (M3)
├── models/
│   ├── __init__.py
│   └── schema.py          # Canonical Pydantic schemas (M1)
├── mcp/
│   └── github_client.py   # BaseGitClient, MockGitClient (Participant 3 interface)
├── rag/                   # Participant 2 (100% complete, 68 passing tests)
│   ├── service.py
│   ├── index.py
│   └── snapshots.py
└── cli/
    └── main.py            # Participant 4 interface
tests/
├── ground_truth/
├── e2e/                   # E2E test suite (M-E2E)
│   ├── conftest.py
│   ├── test_tier1_feature_coverage.py
│   ├── test_tier2_boundaries_corners.py
│   ├── test_tier3_cross_feature.py
│   └── test_tier4_real_world.py
└── unit/
    ├── test_schema.py     # Schema tests (M1)
    ├── test_llm_client.py # LLM client tests (M2)
    ├── test_defense.py    # Defense & prompts tests (M3)
    └── test_orchestrator.py # Orchestrator unit tests (M4)
```
