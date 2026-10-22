from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import date

QA_VERSION = 3

DIGITS = dict(zip("零〇一二三四五六七八九", [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
DIGITS["两"] = 2
UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def chinese_number(value: str) -> str:
    if not any(char in UNITS for char in value):
        return "".join(str(DIGITS[char]) for char in value)
    total = section = number = 0
    for char in value:
        if char in DIGITS:
            number = DIGITS[char]
        else:
            unit = UNITS[char]
            if unit < 10000:
                section += (number or 1) * unit
            elif unit == 10000:
                total += (section + number) * unit
                section = 0
            else:
                total = (total + section + number) * unit
                section = 0
            number = 0
    return str(total + section + number)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"<\|[^>]*\|>", "", value)
    value = re.sub(r"零下(?=[零〇一二三四五六七八九两十百千万亿\d])", "负", value)
    def calendar(match):
        year, month, day = map(int, match.groups())
        try:
            date(year, month, day)
        except ValueError:
            return match[0]
        return f"{year}年{month}月{day}日"
    value = re.sub(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", calendar, value)
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = re.sub(r"([+−-]?\d+(?:\.\d+)?)%", r"百分之\1", value)
    value = value.replace("负百分之", "百分之负").replace("百分之", "\ue000")
    value = re.sub(r"[零〇一二三四五六七八九两十百千万亿]+", lambda m: chinese_number(m[0]), value)
    value = value.replace("\ue000", "百分之")
    value = re.sub(r"(\d{4})年0*(\d{1,2})月0*(\d{1,2})[日号]",
                   lambda m: f"{int(m[1])}年{int(m[2])}月{int(m[3])}日", value)
    def clock(match):
        hour, minute, second = match[1], match[2], match[3]
        if int(hour) > 23 or int(minute) > 59 or (second is not None and int(second) > 59):
            return match[0]
        return f"{int(hour)}时{int(minute)}分" + (f"{int(second)}秒" if second is not None else "")
    value = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", clock, value)
    value = re.sub(r"(\d{1,2})[点时]半", r"\1时30分", value)
    value = re.sub(r"(\d{1,2})[点时]整", r"\1时0分", value)
    value = re.sub(r"(\d{1,2})[点时]0*(\d{1,2})分(?:0*(\d{1,2})秒)?", clock, value)
    value = re.sub(r"((?:凌晨|早上|上午|中午|下午|傍晚|晚上|夜里))(\d{1,2})点(\d{1,2})(?![\d分秒])",
                   lambda m: m[1] + f"{int(m[2])}时{int(m[3])}分" if int(m[2]) <= 23 and int(m[3]) <= 59 else m[0], value)
    value = re.sub(r"(?<=\d)\.(?=\d)", "点", value)
    value = re.sub(r"(?:零下|负|(?<![\d])[-−])(?=\d)", "负", value)
    value = re.sub(r"(?:正|(?<![\d])\+)(?=\d)", "", value)
    value = re.sub(r"负百分之(\d+(?:点\d+)?)", r"百分之负\1", value)
    # Keep numeric relationships before dropping punctuation. Otherwise a
    # range/list/fraction can collapse into a completely different integer.
    atom = r"(?:百分之)?负?\d+(?:点\d+)?"
    def fraction(match):
        numerator, denominator = match[1], match[2]
        return denominator + "分之" + numerator
    value = re.sub(rf"({atom})\s*[/⁄∕]\s*({atom})", fraction, value)
    def signed_fraction(match):
        denominator, numerator = match[1], match[2]
        negative = denominator.startswith("负") != numerator.startswith("负")
        return denominator.removeprefix("负") + "分之" + ("负" if negative else "") + numerator.removeprefix("负")
    value = re.sub(rf"({atom})分之({atom})", signed_fraction, value)
    value = re.sub(rf"({atom})\s*(?:[-−–—~]|至|到)\s*(?={atom})", r"\1至", value)
    # Other separators between numbers are significant even when their exact
    # meaning is uncertain. A list marker prevents silent concatenation.
    value = re.sub(r"(?<=\d)[\W_]+(?=(?:百分之)?负?\d)", "\ue001", value)
    return "".join(c for c in value if c.isalnum() or c == "\ue001").replace("\ue001", "、")


def numeric_expressions(text: str) -> list[str]:
    atom = r"(?:百分之)?负?\d+(?:点\d+)?"
    fraction = rf"(?:{atom}分之{atom}|{atom})"
    interval = rf"{fraction}(?:至{fraction})?"
    return re.findall(rf"{interval}(?:、{interval})*", text)


def normalized_critical_terms(terms: list[str]) -> list[tuple[str, str]]:
    result, seen = [], {}
    for term in terms:
        normalized = normalize(term)
        if not normalized:
            raise ValueError(f"关键字规范化后为空：{term!r}")
        if normalized in seen:
            raise ValueError(f"关键字规范化后重复：{seen[normalized]!r}、{term!r}")
        seen[normalized] = term
        result.append((term, normalized))
    return result


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def evaluate(expected: str, recognized: str, duration: float, cfg: dict, rms_db: float = -20) -> dict:
    a, b = normalize(expected), normalize(recognized)
    errors = edit_distance(a, b)
    cer = errors / max(1, len(a))
    diffs, longest_gap = [], 0
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            diffs.append({"type": tag, "expected": a[i:j], "heard": b[k:l]})
        if tag in ("delete", "insert"):
            longest_gap = max(longest_gap, j-i, l-k)
    reasons = []
    # A changed relationship, sign or digit is material even in a long sentence.
    if numeric_expressions(a) != numeric_expressions(b):
        reasons.append("关键数值或正负号不一致")
    critical_differences = [term for term, normalized in normalized_critical_terms(cfg.get("critical_terms", []))
                            if a.count(normalized) != b.count(normalized)]
    if critical_differences:
        reasons.append("疑似关键字差异：" + "、".join(critical_differences))
    if not b:
        reasons.append("未识别到文字")
    if cer > cfg["max_cer"]:
        reasons.append(f"文字差异率 {cer:.1%}")
    if longest_gap >= cfg["max_gap"]:
        reasons.append(f"疑似连续漏读或重复 {longest_gap} 字")
    rate = len(a) / duration if duration > 0 else 0
    if duration <= 0.2 or (len(a) >= 8 and not cfg["min_chars_per_second"] <= rate <= cfg["max_chars_per_second"]):
        reasons.append("音频时长与文本长度不匹配")
    if rms_db < -50:
        reasons.append("音频近乎静音")
    return {"passed": not reasons, "cer": round(cer, 5), "recognized": recognized,
            "reasons": reasons, "differences": diffs, "duration": duration, "rms_db": rms_db}
