"""Prompt Injection Defense and Raw Data Isolation (Milestone M3).

Provides mechanisms to enforce NFR-002, AC-009, FR-009, and FR-013:
- Delimiter escaping and XML sandboxing (`isolate_raw_data`, `escape_delimiters`)
- Semantic wrappers for diff, spec, and user summary payloads
- Heuristic and pattern-based prompt injection detection (`detect_prompt_injection`)
- Context integrity and partial data evaluation (`evaluate_data_integrity`)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from specguard.models.schema import FileChange, GitContext


# ==============================================================================
# Known Delimiters and Jailbreak Tokens
# ==============================================================================

BREAKOUT_TOKENS: List[Tuple[str, str]] = [
    ("[INST]", "[ESCAPED_INST]"),
    ("[/INST]", "[ESCAPED_/INST]"),
    ("<|im_start|>", "[ESCAPED_IM_START]"),
    ("<|im_end|>", "[ESCAPED_IM_END]"),
    ("<|system|>", "[ESCAPED_SYSTEM]"),
    ("<|user|>", "[ESCAPED_USER]"),
    ("<|assistant|>", "[ESCAPED_ASSISTANT]"),
    ("[SYS]", "[ESCAPED_SYS]"),
    ("[/SYS]", "[ESCAPED_/SYS]"),
    ("<<SYS>>", "[ESCAPED_SYS]"),
    ("<</SYS>>", "[ESCAPED_/SYS]"),
    ("<s>", "[ESCAPED_S]"),
    ("</s>", "[ESCAPED_/S]"),
]

# Patterns for prompt injection scanning
INJECTION_PATTERNS: List[Tuple[str, str, str]] = [
    # (Regex pattern, pattern_name, threat_level)
    # 1. English instruction overrides (Fix 3: expanded for optional 'previous/prior' and 'any/all')
    (
        r"(?i)\b(ignore|disregard|forget|bypass)\s+(?:(?:all|any)\s+)?(?:(?:previous|prior|above|system)\s+)?(instructions|prompts|rules|commands|constraints)\b",
        "instruction_override_disregard",
        "HIGH",
    ),
    # 2. Russian instruction overrides (Fix 1: inflected forms and polite imperatives)
    (
        r"(?i)\b(забыть|забудь(?:те)?|игнорировать|игнорируй(?:те)?|пропустить|пропусти(?:те)?|сбросить|сбрось(?:те)?|отменить|отмени(?:те)?|пренебречь|пренебреги(?:те)?)\s+(?:(?:все|любые)\s+)?(?:(?:предыдущие|прошлые|системные|вышеуказанные)\s+)?(?:инструкции|указания|промпты|правила|команды|ограничения|требования)\b",
        "russian_instruction_override",
        "HIGH",
    ),
    # 3. System override keywords (English)
    (
        r"(?i)\b(system\s+override|override\s+system|emergency\s+override|admin\s+override)\b",
        "system_override_keyword",
        "HIGH",
    ),
    # 4. Russian system override and emergency modes (Fix 1)
    (
        r"(?i)\b(системный\s+сбой|аварийный\s+режим|режим[еа]?\s+(?:разработчика|администратора|бога|root))\b",
        "russian_system_override",
        "HIGH",
    ),
    # 5. Verdict manipulation (Fix 3: implemented, not_affected, 0/no/zero violations, mark all requirements as implemented)
    (
        r"(?i)\b(report\s+(?:implemented(?:\s+with\s+(?:0|zero)\s+violations?)?|(?:no|0|zero)\s+violations?|without\s+violations?)|set\s+(?:the\s+)?status\s*[:=]?\s*(?:to|as)?\s*(?:implemented|not_affected)|mark\s+(?:[a-zA-Z]+\s+){0,3}(?:as\s+)?(?:implemented|not_affected))\b",
        "verdict_manipulation_implemented",
        "HIGH",
    ),
    # 6. Russian verdict manipulation (Fix 1: instrumental plural, participles, status assignment)
    (
        r"(?i)\b(?:(?:отметь(?:те)?|пометь(?:те)?|отметить|пометить|считай(?:те)?|считать)\s+(?:(?:все|каждое)\s+)?(?:требовани[ея]\s+)?(?:как\s+|в\s+)?(?:implemented|not_affected|выполненн\w+|выполнены?|реализованн\w+|реализованы?)|отчет[а-я]*\s+без\s+(?:нарушений|замечаний)|(?:установи(?:те)?|присвой(?:те)?|верни(?:те)?)\s+статус(?:\s+в)?\s+(?:implemented|not_affected)|подтверди(?:те)?\s+все\s+требования)\b",
        "russian_verdict_manipulation",
        "HIGH",
    ),
    # 7. Developer mode jailbreak (English)
    (
        r"(?i)\b(you\s+are\s+now\s+in\s+developer\s+mode|dan\s+mode|jailbreak(\s+mode)?)\b",
        "developer_mode_jailbreak",
        "HIGH",
    ),
    # 8. Role hijacking (English & Russian)
    (
        r"(?i)\b(pretend\s+you\s+are|act\s+as\s+(?:an?\s+)?(?:admin|administrator|root|system)|role:\s*system|действуй\s+как\s+(?:администратор|система|root)|притворись\s+(?:администратором|системой|root))\b",
        "role_hijacking",
        "HIGH",
    ),
    # 9. Instruction redefinition (English & Russian)
    (
        r"(?i)\b(new\s+instructions:(?:\s|$)|from\s+now\s+on\s+you\s+must\b|your\s+new\s+task\s+is\s+to\b|новые\s+инструкции:(?:\s|$)|с\s+этого\s+момента\s+ты\s+должен\b)",
        "instruction_redefinition",
        "MEDIUM",
    ),
    # 10. Audit suppression (Fix 3: expanded for implemented/not_affected/pass/success/valid/skip audit)
    (
        r"(?i)\b(always\s+return\s+(?:implemented|not_affected|pass|success|valid)|auto-?approve\s+(?:this\s+)?(?:pr|commit|pull\s+request)|(?:do\s+not|skip)\s+(?:this\s+)?audit|всегда\s+возвращай\s+(?:implemented|not_affected)|не\s+проводи\s+аудит|автоматически\s+одобри\s+этот\s+(?:pr|коммит|pull\s+request))\b",
        "audit_suppression",
        "HIGH",
    ),
    # 11. Raw delimiter breakout (Fix 2: supporting attributes and self-closing tags)
    (
        r"(?i)</?\s*raw_[a-zA-Z0-9_\-]+(?:\s+[^>]*)?/?\s*>",
        "raw_delimiter_breakout",
        "MEDIUM",
    ),
    # 12. ChatML & LLM Token Injection (Fix 3: full parity with all 13 BREAKOUT_TOKENS)
    (
        r"(?i)(\[/?(INST|SYS)\]|<\|im_(start|end)\|>|<\|(system|user|assistant)\|>|<<SYS>>|<</SYS>>|<s>|</s>)",
        "chatml_token_injection",
        "HIGH",
    ),
]


# ==============================================================================
# Delimiter Escaping and XML Sandboxing
# ==============================================================================

def escape_delimiters(content: str) -> str:
    """Neutralize boundary breakout tags and known LLM delimiter tokens.

    Replaces closing or opening `<raw_...>` tags with safe tokens and neutralizes
    special tokens like `[INST]`, `[/INST]`, `<|im_start|>`, `<|im_end|>`.

    Args:
        content: Raw string that may contain untrusted delimiters.

    Returns:
        Sanitized string safe for encapsulation inside system prompt XML blocks.
    """
    if not content:
        return ""

    safe = content

    # 1. Escape any <raw_...> or </raw_...> tags (case-insensitive, with attributes/self-closing)
    safe = re.sub(
        r"</?\s*raw_([a-zA-Z0-9_\-]+)(?:\s+[^>]*)?/?\s*>",
        r"[ESCAPED_TAG:raw_\1]",
        safe,
        flags=re.IGNORECASE,
    )

    # 2. Escape known special instruction tokens (case-insensitive regex replace)
    for token, replacement in BREAKOUT_TOKENS:
        safe = re.sub(re.escape(token), replacement, safe, flags=re.IGNORECASE)

    return safe


def isolate_raw_data(tag: str, content: Optional[str]) -> str:
    """Safely encapsulate untrusted data within XML-style boundary tags (NFR-002).

    Escapes any closing boundary tags and known prompt injection delimiter tokens
    inside the content to prevent jailbreaking or delimiter breakouts.

    Args:
        tag: Semantic boundary name (e.g. 'diff', 'spec', 'user_summary', 'user_comment').
        content: Raw untrusted string content from repository, spec, or user.

    Returns:
        Escaped content cleanly enclosed in `<raw_{tag}>\\n...\\n</raw_{tag}>`.
    """
    safe_content = escape_delimiters(content or "")

    # Further ensure tag-specific breakout tags are strictly neutralized (including attributes/self-closing)
    safe_content = re.sub(
        rf"</?\s*raw_{re.escape(tag)}(?:\s+[^>]*)?/?\s*>",
        f"[ESCAPED_TAG:raw_{tag}]",
        safe_content,
        flags=re.IGNORECASE,
    )
    safe_content = re.sub(
        rf"</?\s*{re.escape(tag)}(?:\s+[^>]*)?/?\s*>",
        f"[ESCAPED_TAG:{tag}]",
        safe_content,
        flags=re.IGNORECASE,
    )

    return f"<raw_{tag}>\n{safe_content}\n</raw_{tag}>"


# ==============================================================================
# Semantic Wrappers
# ==============================================================================

def wrap_diff_for_prompt(diff_text: Optional[str]) -> str:
    """Wrap raw Git diff content in safe `<raw_diff>` boundary tags."""
    return isolate_raw_data("diff", diff_text)


def wrap_spec_for_prompt(spec_text: Optional[str]) -> str:
    """Wrap raw technical specification text in safe `<raw_spec>` boundary tags."""
    return isolate_raw_data("spec", spec_text)


def wrap_user_summary_for_prompt(summary: Optional[str]) -> str:
    """Wrap user summary text in safe `<raw_user_summary>` boundary tags."""
    return isolate_raw_data("user_summary", summary)


def wrap_user_comment_for_prompt(comment: Optional[str]) -> str:
    """Wrap user dispute comment in safe `<raw_user_comment>` boundary tags."""
    return isolate_raw_data("user_comment", comment)


# ==============================================================================
# Injection Scanner
# ==============================================================================

def detect_prompt_injection(text: str) -> Dict[str, Any]:
    """Scan untrusted input text for known prompt injection patterns (NFR-002, AC-009).

    Identifies instruction override attempts, canary phrases, role hijacking,
    and chat delimiter attacks.

    Args:
        text: Untrusted text to analyze (e.g., commit diff, PR comment, user summary).

    Returns:
        Dictionary with analysis results:
        - `detected`: bool, True if any injection pattern matched.
        - `is_injection`: bool, alias for `detected`.
        - `threat_level`: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'.
        - `patterns_matched`: List[str] of matched pattern names.
        - `details`: List[str] of human-readable findings suitable for limitations.
    """
    if not text:
        return {
            "detected": False,
            "is_injection": False,
            "threat_level": "NONE",
            "patterns_matched": [],
            "details": [],
        }

    matched_patterns: List[str] = []
    details: List[str] = []
    max_threat = "NONE"

    threat_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    for pattern_regex, name, level in INJECTION_PATTERNS:
        match = re.search(pattern_regex, text)
        if match:
            matched_patterns.append(name)
            snippet = match.group(0).strip()
            details.append(
                f"Detected suspicious pattern '{name}' (level: {level}): '{snippet}'"
            )
            if threat_rank[level] > threat_rank[max_threat]:
                max_threat = level

    is_inj = len(matched_patterns) > 0
    return {
        "detected": is_inj,
        "is_injection": is_inj,
        "threat_level": max_threat,
        "patterns_matched": matched_patterns,
        "details": details,
    }


# ==============================================================================
# Data Integrity & Partial Data Evaluation
# ==============================================================================

def evaluate_data_integrity(
    git_context: GitContext,
    max_diff_chars: int = 500_000,
) -> Tuple[bool, List[str], List[str]]:
    """Evaluate GitContext integrity and detect partial or truncated inputs (FR-009, FR-013).

    Examines the GitContext for:
    1. Pre-existing partial markers (`is_partial == True`).
    2. Total diff character overflow exceeding `max_diff_chars`.
    3. Individual files with `is_truncated == True`.
    4. Missing patches on modified/added files.
    5. Merge commits lacking `first_parent_sha` (AC-014).

    Args:
        git_context: The GitContext collected by git/MCP adapters.
        max_diff_chars: Maximum allowable character count across all diff patches.

    Returns:
        Tuple of `(is_partial: bool, limitations: List[str], missing_data: List[str])`:
        - `is_partial`: True if any data limitation or truncation was detected.
        - `limitations`: Descriptive limitation notes (deduplicated).
        - `missing_data`: Machine-readable identifiers of missing data items.
    """
    is_partial = bool(git_context.is_partial)
    limitations: List[str] = list(git_context.limitations)
    missing_data: List[str] = []

    # 1. Check for Merge Commit First-Parent Lineage (AC-014, FR-009)
    if git_context.is_merge_commit and not git_context.first_parent_sha:
        is_partial = True
        limitations.append(
            "Merge commit is missing first_parent_sha; lineage comparison is partial."
        )
        missing_data.append("first_parent_sha")

    # 2. Check File-Level Truncation and Missing Patches
    total_chars = 0
    for f in git_context.files:
        patch_len = len(f.patch) if f.patch else 0
        total_chars += patch_len

        if f.is_truncated:
            is_partial = True
            limitations.append(f"File diff is truncated: {f.filename}")
            missing_data.append(f"truncated:{f.filename}")

        if f.patch is None and f.status.lower() not in (
            "deleted",
            "removed",
            "binary",
            "unmodified",
        ):
            is_partial = True
            limitations.append(f"File patch unavailable: {f.filename}")
            missing_data.append(f"missing_patch:{f.filename}")

    # 3. Check Overall Diff Character Threshold Overflow
    if max_diff_chars > 0 and total_chars > max_diff_chars:
        is_partial = True
        limitations.append(
            f"Total diff size ({total_chars} chars) exceeds maximum threshold "
            f"({max_diff_chars} chars); analysis is partial."
        )
        missing_data.append("diff_character_overflow")

    # Deduplicate while preserving insertion order
    def _dedup(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    return is_partial, _dedup(limitations), _dedup(missing_data)


__all__ = [
    "BREAKOUT_TOKENS",
    "INJECTION_PATTERNS",
    "escape_delimiters",
    "isolate_raw_data",
    "wrap_diff_for_prompt",
    "wrap_spec_for_prompt",
    "wrap_user_summary_for_prompt",
    "wrap_user_comment_for_prompt",
    "detect_prompt_injection",
    "evaluate_data_integrity",
]
