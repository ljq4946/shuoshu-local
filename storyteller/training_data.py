"""Read-only source audit and reproducible splits for local voice fine-tuning."""
from __future__ import annotations

import array
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path
from .storage import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_VERSION = 1
# Conservative text screening; excluded rows remain documented, never deleted.
CONTENT_RE = re.compile(
    r"セックス|性交|精液|射精|勃起|乳首|乳房|膣|陰部|肉棒|チン[ポコ]|"
    r"クリトリス|オナニー|手コキ|フェラ|中出し|おちん|ペニス|ヴァギナ|"
    r"愛液|パンティ|下着|裸|挿[入れ]|エッチ|えっち|濡れ|舐め|"
    r"揉[むんみ]|喘[ぎぐ]|イク[っッ！!]|イ[ッっ]ちゃ|気持ち[い良]|"
    r"おっぱい|おまん|快感|欲情|淫|性器"
)
NONVERBAL_RE = re.compile(r"^[…・、。！？!?ー〜～っッぁァあアぅゥうウんンふフむムはハくクひヒへヘえエおオほホぃィいイ\s]+$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, value) -> None:
    write_json(path, value)


def dump_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


def resolve_audio(old_path: str, audio_root: Path) -> Path:
    # GPT-SoVITS labels commonly retain another machine's absolute path.
    name = old_path.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name in (".", "..") or ":" in name:
        raise ValueError("Invalid audio basename")
    candidate = (audio_root / name).resolve()
    if candidate.parent != audio_root.resolve():
        raise ValueError("Audio path escapes the supplied directory")
    return candidate


def audio_metadata(path: Path) -> dict:
    with wave.open(str(path), "rb") as source:
        channels, width, rate, frames = source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()
        if width != 2 or source.getcomptype() != "NONE":
            raise ValueError("Training source must be uncompressed 16-bit PCM")
        pcm = source.readframes(frames)
    if len(pcm) != frames * channels * width or not frames:
        raise ValueError("Truncated or empty WAV")
    samples = array.array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max(abs(int(s)) for s in samples) / 32768
    rms = math.sqrt(sum(float(s) * s for s in samples) / len(samples)) / 32768
    header = f"{channels}:{width}:{rate}:".encode("ascii")
    return {
        "sample_rate": rate, "channels": channels, "sample_width": width,
        "duration": frames / rate, "peak": peak, "rms": rms,
        "clipped_fraction": sum(abs(int(s)) >= 32760 for s in samples) / len(samples),
        "pcm_sha256": hashlib.sha256(header + pcm).hexdigest(), "audio_sha256": sha256(path),
    }


def assign_splits(rows: list[dict], seed: int = 20261022) -> dict[str, list[dict]]:
    """Connected components keep identical text AND identical audio together."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in ("text:" + row["text_key"], "pcm:" + row["pcm_sha256"]):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    groups = sorted(groups.values(), key=lambda group: min(row["id"] for row in group))
    random.Random(seed).shuffle(groups)
    splits = {"train": [], "validation": [], "test": []}
    target = max(1, round(len(rows) * 0.05))
    for group in groups:
        destination = "validation" if len(splits["validation"]) < target else "test" if len(splits["test"]) < target else "train"
        splits[destination].extend(group)
    for split, items in splits.items():
        items.sort(key=lambda row: row["id"])
        for row in items:
            row["split"] = split
    for i, a in enumerate(splits):
        for b in list(splits)[i + 1:]:
            for key in ("text_key", "pcm_sha256", "id"):
                if {r[key] for r in splits[a]} & {r[key] for r in splits[b]}:
                    raise RuntimeError(f"Cross-split leakage: {a}/{b}/{key}")
    return splits


def prepare_dataset(audio_root: Path, labels: Path, output: Path, seed: int = 20261022) -> dict:
    audio_root, labels, output = audio_root.resolve(), labels.resolve(), output.resolve()
    if output == audio_root or output == labels.parent or audio_root in output.parents:
        raise ValueError("Output must be separate from the supplied source directories")
    if (output / "features").exists() or (output / "runs").exists():
        raise ValueError("A prepared dataset with features/runs is immutable; use a new output directory")
    rows, excluded, source_records = [], [], []
    seen_ids, seen_pcm = set(), set()
    for line_no, line in enumerate(labels.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ValueError(f"Malformed label line {line_no}")
        old_path, speaker, language, text = fields
        path = resolve_audio(old_path, audio_root)
        meta = audio_metadata(path)
        key = normalized_text(text)
        row = {"id": path.stem, "audio": str(path), "speaker": speaker, "language": language.upper(),
               "text": text.strip(), "text_key": key, "source_line": line_no, **meta}
        source_records.append({"id": row["id"], "audio_sha256": meta["audio_sha256"], "source_line": line_no})
        reasons = []
        if row["id"] in seen_ids:
            reasons.append("duplicate_id")
        if meta["pcm_sha256"] in seen_pcm:
            reasons.append("duplicate_pcm")
        seen_ids.add(row["id"])
        seen_pcm.add(meta["pcm_sha256"])
        if language.upper() != "JA" or not re.search(r"[\u3040-\u30ff]", text):
            reasons.append("not_japanese_transcription")
        if CONTENT_RE.search(text):
            reasons.append("content_screen")
        if NONVERBAL_RE.fullmatch(text.strip()):
            reasons.append("nonverbal_only")
        if len(key) < 4:
            reasons.append("too_short_text")
        if not 1.0 <= meta["duration"] <= 12.0:
            reasons.append("duration_out_of_range")
        if meta["rms"] < 0.002 or meta["clipped_fraction"] > 0.01:
            reasons.append("audio_quality")
        if reasons:
            excluded.append({**row, "reasons": reasons})
        else:
            rows.append(row)
    if len(rows) < 100:
        raise ValueError(f"Only {len(rows)} usable rows; supplement or review the source data")
    splits = assign_splits(rows, seed)
    label_sha = sha256(labels)
    source_fingerprint = hashlib.sha256(json.dumps([label_sha, source_records], sort_keys=True).encode()).hexdigest()
    for split, items in splits.items():
        dump_jsonl(output / f"{split}.jsonl", items)
    dump_jsonl(output / "excluded.jsonl", excluded)
    dump_json(output / "sources.json", {"labels": str(labels), "labels_sha256": label_sha, "audio_root": str(audio_root), "files": source_records})
    report = {
        "version": DATA_VERSION, "seed": seed, "source_fingerprint": source_fingerprint,
        "source_rows": len(source_records), "source_wav_files": len(list(audio_root.glob("*.wav"))),
        "source_hours": sum(row["duration"] for row in rows + excluded) / 3600,
        "usable_rows": len(rows), "usable_hours": sum(row["duration"] for row in rows) / 3600,
        "excluded_rows": len(excluded), "exclusion_reasons": dict(Counter(reason for row in excluded for reason in row["reasons"])),
        "splits": {s: {"rows": len(items), "hours": sum(r["duration"] for r in items) / 3600} for s, items in splits.items()},
        "manifest_sha256": {s: sha256(output / f"{s}.jsonl") for s in splits},
        "cross_split_exact_text_leaks": 0, "cross_split_pcm_leaks": 0,
        "source_modified": False, "supplement_used": False,
        "task": "Japanese paired voice adaptation; Chinese synthesis retains the multilingual base model.",
        "limitations": ["No line-by-line human listening/alignment verification", "No paired Chinese speech in this dataset",
                        "Exact text/PCM grouping prevents exact leakage; related dialogue can still share scenes"],
    }
    dump_json(output / "dataset_report.json", report)
    return report


def verify_sources(dataset: Path) -> dict:
    sources = json.loads((dataset / "sources.json").read_text(encoding="utf-8"))
    if sha256(Path(sources["labels"])) != sources["labels_sha256"]:
        raise ValueError("Source labels changed since preparation")
    for row in sources["files"]:
        path = Path(sources["audio_root"]) / (row["id"] + ".wav")
        if sha256(path) != row["audio_sha256"]:
            raise ValueError(f"Source audio changed: {row['id']}")
    return {"labels_unchanged": True, "audio_files_verified": len(sources["files"]), "audio_unchanged": True}
