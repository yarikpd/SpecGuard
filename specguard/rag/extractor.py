import re
from typing import List, Tuple

from specguard.models.schema import SpecRequirement

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
ID_PATTERN = re.compile(r"\b([A-Z]{2,6}-\d+)\b", re.IGNORECASE)
RANGE_PATTERN = re.compile(
    r"\b([A-Z]{2,6})-(\d+)\s*[–—\-]\s*(?:([A-Z]{2,6})-)?(\d+)\b",
    re.IGNORECASE,
)
ITEM_START_EXPLICIT_RE = re.compile(
    r"^(?:[-*+]\s+|\d+[.)]\s+)?(?:\*{1,2}|_{1,2})?(?:\[)?([A-Z]{2,6}-\d+)(?:\])?(?:\*{1,2}|_{1,2})?[:.\s-]",
    re.IGNORECASE,
)
LIST_ITEM_START_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)\S")

COND_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"если(?:\s+и\s+только\s+если)?|"
    r"при\s+условии(?:\s*,?\s*что)?|"
    r"в\s+случае(?:\s*,?\s*если|\s*,?\s*когда)?|"
    r"при\s+наличии|"
    r"только\s+если|"
    r"if|"
    r"when|"
    r"provided\s+that|"
    r"as\s+long\s+as"
    r")\b",
    re.IGNORECASE,
)

EXC_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"кроме(?:\s+тех\s+случаев(?:\s*,?\s*когда)?|\s+случаев(?:\s*,?\s*когда)?|\s+как)?|"
    r"за\s+исключением(?:\s+случаев(?:\s*,?\s*когда)?)?|"
    r"не\s+распространяется(?:\s+на)?|"
    r"исключая|"
    r"исключение(?::|\s*-)?|"
    r"except(?:\s+for|\s+when)?|"
    r"unless|"
    r"with\s+the\s+exception\s+of|"
    r"excluding"
    r")\b",
    re.IGNORECASE,
)


def _clean_description(raw_text: str, req_id: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", text).strip()
    escaped_id = re.escape(req_id)
    text = re.sub(
        rf"^(?:\*{1,2}|_{1,2})?(?:\[)?{escaped_id}(?:\])?(?:\*{1,2}|_{1,2})?[:.\s-]*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def _make_title(description: str, max_len: int = 80) -> str:
    cleaned = description.strip()
    if not cleaned:
        return ""

    colon_idx = cleaned.find(":")
    if 0 < colon_idx <= 50 and not cleaned.startswith("["):
        lead = cleaned[:colon_idx].strip()
        if len(lead) >= 3 and "\n" not in lead:
            return lead

    first_sentence = re.split(r"[.!?\n]", cleaned)[0].strip()
    if first_sentence and len(first_sentence) <= max_len:
        return first_sentence

    if len(cleaned) <= max_len:
        return cleaned

    truncated = cleaned[:max_len]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + "..."


def _extract_conditions_and_exceptions(text: str) -> Tuple[List[str], List[str]]:
    conditions: List[str] = []
    exceptions: List[str] = []

    sentences = re.split(r"(?<=[.!?\n;])\s+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        triggers: List[Tuple[int, int, str, str]] = []
        for m in COND_TRIGGER_RE.finditer(sentence):
            triggers.append((m.start(), m.end(), "cond", m.group(0)))
        for m in EXC_TRIGGER_RE.finditer(sentence):
            triggers.append((m.start(), m.end(), "exc", m.group(0)))

        if not triggers:
            continue

        triggers.sort(key=lambda x: x[0])

        filtered_triggers: List[Tuple[int, int, str, str]] = []
        last_end = -1
        for t in triggers:
            if t[0] >= last_end:
                filtered_triggers.append(t)
                last_end = t[1]

        for i, (start, end, kind, _) in enumerate(filtered_triggers):
            if i + 1 < len(filtered_triggers):
                clause_end = filtered_triggers[i + 1][0]
            else:
                clause_end = len(sentence)

            clause = sentence[start:clause_end].strip()
            clause = re.sub(r"[,;:.!?\s]+$", "", clause).strip()
            if not clause:
                continue

            if kind == "cond":
                if clause not in conditions:
                    conditions.append(clause)
            else:
                if clause not in exceptions:
                    exceptions.append(clause)

    return conditions, exceptions


def _extract_references(text: str, current_req_id: str) -> List[str]:
    refs: List[str] = []
    seen = {current_req_id.upper()}

    for m in RANGE_PATTERN.finditer(text):
        p1, start_s, p2, end_s = m.group(1), m.group(2), m.group(3), m.group(4)
        prefix = p1.upper()
        if p2 and p2.upper() != prefix:
            continue
        try:
            start_num = int(start_s)
            end_num = int(end_s)
            if 0 <= end_num - start_num <= 50:
                width = max(len(start_s), len(end_s), 3)
                for num in range(start_num, end_num + 1):
                    rid = f"{prefix}-{num:0{width}d}"
                    if rid not in seen:
                        seen.add(rid)
                        refs.append(rid)
        except ValueError:
            pass

    for m in ID_PATTERN.finditer(text):
        rid = m.group(1).upper()
        if rid not in seen:
            seen.add(rid)
            refs.append(rid)

    return refs


class RequirementExtractor:
    def extract(self, text: str, source_file: str = "") -> List[SpecRequirement]:
        if not text or not text.strip():
            return []

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")

        has_explicit_id = bool(
            re.search(
                r"(?m)^(?:\s*[-*+]\s+|\s*\d+[.)]\s+)?(?:\*{0,2}|_{0,2})(?:\[)?[A-Z]{2,6}-\d+",
                normalized,
                re.IGNORECASE,
            )
        )
        has_list_items = bool(re.search(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)\S", normalized))

        section_stack: List[Tuple[int, str]] = []
        raw_items: List[Tuple[str, str]] = []

        current_item_lines: List[str] = []
        current_item_section = ""

        def flush_item():
            nonlocal current_item_lines, current_item_section
            if current_item_lines:
                item_text = "\n".join(current_item_lines).strip()
                if item_text:
                    raw_items.append((item_text, current_item_section))
                current_item_lines = []

        def get_current_section() -> str:
            return " > ".join(title for _, title in section_stack)

        def update_heading(level: int, title: str):
            flush_item()
            cleaned_title = title.strip().rstrip("#").strip()
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, cleaned_title))

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            line_stripped = line.strip()

            if line_stripped and i + 1 < n:
                next_line = lines[i + 1].strip()
                if re.match(r"^={3,}\s*$", next_line):
                    update_heading(1, line_stripped)
                    i += 2
                    continue
                elif re.match(r"^-{3,}\s*$", next_line):
                    update_heading(2, line_stripped)
                    i += 2
                    continue

            heading_match = HEADING_RE.match(line_stripped)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                update_heading(level, title)
                i += 1
                continue

            is_explicit_start = bool(ITEM_START_EXPLICIT_RE.match(line_stripped))
            is_list_start = bool(LIST_ITEM_START_RE.match(line_stripped))

            if has_explicit_id:
                if is_explicit_start or is_list_start:
                    flush_item()
                    current_item_section = get_current_section()
                    current_item_lines.append(line)
                elif current_item_lines:
                    if not line_stripped:
                        flush_item()
                    else:
                        current_item_lines.append(line)
            elif has_list_items:
                if is_list_start:
                    flush_item()
                    current_item_section = get_current_section()
                    current_item_lines.append(line)
                elif current_item_lines:
                    if not line_stripped:
                        flush_item()
                    else:
                        current_item_lines.append(line)
            else:
                if not line_stripped:
                    flush_item()
                else:
                    if not current_item_lines:
                        current_item_section = get_current_section()
                    current_item_lines.append(line)

            i += 1

        flush_item()

        existing_ids = set()
        for raw_text, _ in raw_items:
            m = ITEM_START_EXPLICIT_RE.match(raw_text.strip())
            if m:
                existing_ids.add(m.group(1).upper())

        counter = 1
        requirements: List[SpecRequirement] = []

        for raw_text, section in raw_items:
            m = ITEM_START_EXPLICIT_RE.match(raw_text.strip())
            if m:
                req_id = m.group(1).upper()
            else:
                while f"REQ-{counter:03d}" in existing_ids:
                    counter += 1
                req_id = f"REQ-{counter:03d}"
                existing_ids.add(req_id)
                counter += 1

            description = _clean_description(raw_text, req_id)
            title = _make_title(description)
            conditions, exceptions = _extract_conditions_and_exceptions(description)
            related = _extract_references(raw_text, req_id)

            requirements.append(
                SpecRequirement(
                    req_id=req_id,
                    section=section,
                    title=title,
                    description=description,
                    conditions=conditions,
                    exceptions=exceptions,
                    raw_text=raw_text,
                    source_file=source_file,
                    related_req_ids=related,
                )
            )

        return requirements
