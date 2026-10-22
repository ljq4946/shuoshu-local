from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path

HEADING = re.compile(
    r"^(?:第[零〇一二三四五六七八九十百千万两\d]+"
    r"(?:[章卷].*|[回节部集篇](?:\s.*|[：:、.．].*|$))"
    r"|chapter\s+\d+.*|(?:序章|序言|楔子|引子|尾声|后记|终章)(?:\s.*|[：:].*|$))$", re.I)
ANNOTATION = re.compile(r"<([^<>|]+)\|([^<>|]+)>")
NUMBER = re.compile(r"[+−-]?\d+(?:[.,，．:：/\-]\d+)*%?")


def is_structure(text: str) -> bool:
    return bool(text.strip()) and not any(char.isalnum() for char in text)


def is_heading(line: str, pattern: str | None = None) -> bool:
    if pattern is not None:
        return bool(re.fullmatch(pattern, line))
    # 回/节/部/集/篇 also count ordinary events and objects, so require an
    # explicit title boundary. Compact 章/卷 titles remain supported; unusual
    # manuscript conventions can use an explicit pattern or source line list.
    return (len(line) <= 80 and not re.search(r"[。，,！？!?；;…]|\.{2,}", line)
            and bool(HEADING.fullmatch(line)))


def protected_spans(text: str, protected_words=()) -> list[tuple[int, int]]:
    patterns = [ANNOTATION, NUMBER]
    if protected_words:
        patterns.append(re.compile('|'.join(re.escape(word) for word in
                                            sorted(protected_words, key=len, reverse=True))))
    spans = sorted((match.start(), match.end()) for pattern in patterns for match in pattern.finditer(text))
    merged = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def align_segment_boundaries(parts: list[str], protected_words=()) -> list[str]:
    """Move only boundaries inside a protected item; retain all other cuts."""
    text = ''.join(parts)
    spans = protected_spans(text, protected_words)
    starts = [start for start, _ in spans]
    cuts, position = {0, len(text)}, 0
    for part in parts[:-1]:
        position += len(part)
        cut = position
        index = bisect_right(starts, cut) - 1
        if index >= 0:
            start, end = spans[index]
            if start < cut < end:
                cut = start if start else end
        cuts.add(cut)
    cuts = sorted(cuts)
    return [text[start:end] for start, end in zip(cuts, cuts[1:]) if end > start]


def split_inference_text(text: str, max_chars: int = 40) -> list[str]:
    """Keep numbers and pronunciation annotations intact in low-VRAM chunks.

    Match upstream's punctuation-based memory limit: a clause without a safe
    punctuation boundary may exceed max_chars. Token splitting then happens
    after the text frontend has normalized complete numbers.
    """
    spans = protected_spans(text)
    cuts = [match.end() for match in re.finditer(r"[，。！？、；：,.!?;:\n]", text)
            if not any(start < match.end() < end for start, end in spans)]
    parts, start = [], 0
    for end in [*cuts, len(text)]:
        if end > start:
            parts.append(text[start:end])
            start = end
    segments, current = [], ""
    for part in parts:
        if current and len(current) + len(part) > max_chars:
            segments.append(current)
            current = ""
        current += part
    if current:
        segments.append(current)
    return segments


def read_book(path: Path, encoding: str | None = None) -> tuple[str, str]:
    raw = path.read_bytes()
    encodings = [encoding] if encoding else ["utf-8-sig", "gb18030"]
    for candidate in encodings:
        try:
            result = raw.decode(candidate)
            if "\x00" in result:
                raise ValueError("书稿含 NUL 字节，请转换成 UTF-8 TXT。")
            return result.replace("\r\n", "\n").replace("\r", "\n"), candidate
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别书稿编码，请使用 --encoding 指定或另存为 UTF-8 TXT。")


def split_chapters(text: str, title: str, pattern: str | None = None,
                   starts: list[int] | None = None) -> list[dict]:
    chapters, lines, heading = [], [], title
    source_lines = text.splitlines()
    if pattern is not None:
        re.compile(pattern)
    if starts is not None:
        if any(type(item) is not int or not 1 <= item <= len(source_lines)
               or not source_lines[item-1].strip() for item in starts):
            raise ValueError("章节起始行必须是原稿中非空行的行号。")
        starts = set(starts)
    for number, line in enumerate(source_lines, 1):
        stripped = line.strip()
        if stripped and (number in starts if starts is not None else is_heading(stripped, pattern)):
            if any(part.strip() for part in lines):
                chapters.append({"title": heading, "text": "\n".join(lines).strip()})
            heading, lines = stripped, [stripped]
        else:
            lines.append(line)
    if any(part.strip() for part in lines):
        chapters.append({"title": heading, "text": "\n".join(lines).strip()})
    if not chapters:
        raise ValueError("书稿为空。")
    return chapters


def split_segments(text: str, max_chars: int = 180, protected_words=()) -> list[str]:
    """Keep every non-whitespace character; prefer sentences, then clauses."""
    result = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        first = len(result)
        sentences = re.findall(r"[^。！？!?；;]+[。！？!?；;]*[”’\"」』）)]*|[。！？!?；;]+", paragraph)
        pending = ""
        for sentence in sentences:
            while len(sentence) > max_chars:
                if pending:
                    result.append(pending)
                    pending = ""
                cut = max((sentence.rfind(mark, 0, max_chars) + 1 for mark in "，,、：: \t"), default=0)
                if cut < max_chars // 3:
                    cut = max_chars
                result.append(sentence[:cut])
                sentence = sentence[cut:]
            if pending and len(pending) + len(sentence) > max_chars:
                result.append(pending)
                pending = ""
            pending += sentence
        if pending:
            result.append(pending)
        result[first:] = align_segment_boundaries(result[first:], protected_words)
    compact = lambda value: re.sub(r"\s+", "", value)
    if compact("".join(result)) != compact(text):
        raise ValueError("分段校验失败，文本发生丢失；请保留书稿并报告此问题。")
    return [part for part in result if compact(part)]


def validate_pronunciations(rules: dict) -> None:
    if not isinstance(rules, dict):
        raise ValueError("读音表必须是 JSON 对象。")
    for word, spoken in rules.items():
        if not isinstance(word, str) or not word or not isinstance(spoken, str):
            raise ValueError("读音表必须使用非空词语和字符串标注。")
        original = ANNOTATION.sub(lambda match: match[1], spoken)
        if original != word or not ANNOTATION.search(spoken):
            raise ValueError(f"读音表不能改写原文，示例：重庆 → <重|CHONG2>庆；问题词：{word}")
        if re.search(r"[<>|]", ANNOTATION.sub("", spoken)):
            raise ValueError(f"读音标注格式错误：{word}")


def apply_pronunciations(text: str, rules: dict) -> str:
    validate_pronunciations(rules)
    if not rules:
        return text
    pattern = re.compile("|".join(re.escape(word) for word in sorted(rules, key=len, reverse=True)))
    return pattern.sub(lambda match: rules[match[0]], text)
