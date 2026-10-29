# E2E Test Infra: SpecGuard (Participant 1)

## Test Philosophy
- Opaque-box, requirement-driven. Derived directly from user requirements (`ORIGINAL_REQUEST.md`, `specification/`), independent of implementation design.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Interaction Testing + Real-World Workload Testing.
- 100% Offline execution without external API keys or live network calls (R5).

## Feature Inventory & Test Mapping
| # | Feature | Source (Requirement) | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Pairwise) |
|---|---------|----------------------|:-----------------:|:-----------------:|:-----------------:|
| 1 | Packaging & Schema Contracts | R1, ARCHITECTURE.md, schema.py | ≥5 | ≥5 | ✓ |
| 2 | LLM Client & Mock Engine | R2, R5, FR-007, AC-004 | ≥5 | ≥5 | ✓ |
| 3 | Prompt Injection Defense | R3, NFR-002, AC-009 | ≥5 | ≥5 | ✓ |
| 4 | Prompts & Scoring Rules | R2, FR-004, FR-008 | ≥5 | ≥5 | ✓ |
| 5 | Core Audit Orchestration | R1, FR-003, AC-002 | ≥5 | ≥5 | ✓ |
| 6 | Merge Commit First-Parent Tracking | R1, TEST-002, AC-014 | ≥5 | ≥5 | ✓ |
| 7 | Partial Data & Truncation Handling | R3, FR-009, FR-013 | ≥5 | ≥5 | ✓ |
| 8 | Dispute Review Procedure | R4, FR-012, AC-008 | ≥5 | ≥5 | ✓ |
| 9 | Inaccessible Base Handling | AC-015 | ≥5 | ≥5 | ✓ |

## Test Architecture
- Test runner: `pytest tests/e2e/ -v`
- Pass/fail semantics: 100% pass with exit code 0
- Offline execution: utilizes `MockGitClient` and `MockLLMClient`
- Directory layout:
  ```
  tests/e2e/
  ├── conftest.py
  ├── test_tier1_feature_coverage.py
  ├── test_tier2_boundaries_corners.py
  ├── test_tier3_cross_feature.py
  └── test_tier4_real_world.py
  ```

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Standard Feature PR Audit & Implementation Acceptance | F1, F2, F4, F5 | Medium |
| 2 | Confirmed Security Violation Detection & Evidence Pinpointing | F4, F5, F6 | Medium |
| 3 | Merge Commit Audit with First-Parent Lineage Resolution | F5, F6, F7 | Medium |
| 4 | Truncated Diff & Inaccessible Base Degradation Audit | F5, F7, F9 | High |
| 5 | Hermetic Dispute Review with Legitimate Overturn Arbitration | F1, F5, F8 | High |
| 6 | Adversarial Multi-Vector Prompt Injection Defense in PR & Dispute | F3, F4, F5, F8 | High |

## Coverage Thresholds
- Tier 1: ≥5 per feature (9 features × 5 = ≥45 tests)
- Tier 2: ≥5 per feature (where boundaries exist, ≥45 tests)
- Tier 3: pairwise coverage of major feature interactions (≥10 tests)
- Tier 4: ≥6 realistic application scenarios (≥6 tests)
- Total E2E Target: ≥106 test cases
