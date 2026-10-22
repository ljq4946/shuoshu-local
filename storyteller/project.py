from __future__ import annotations

import shutil
import tempfile
import uuid
from collections import defaultdict, deque
from pathlib import Path

from .audio import prepare_reference, wav_info
from .config import book_inference_gap, book_settings, generation_profile, inference_gap
from .quality import QA_VERSION
from .storage import contained, digest, file_hash, now, read_json, write_json
from .text import (align_segment_boundaries, apply_pronunciations, read_book,
                   is_structure, split_chapters, split_segments, validate_pronunciations)

ACCEPTED = {"passed", "approved", "pause"}


def save(root: Path, project: dict) -> None:
    for output in project.pop("stale_exports", []):
        stamp = uuid.uuid4().hex[:10]
        for key in ("wav", "mp3"):
            source = contained(root / "exports", output[key])
            if source.is_file():
                history = root / "exports/history"
                history.mkdir(exist_ok=True)
                source.replace(history / f"{source.stem}_{stamp}{source.suffix}")
    project["updated_at"] = now()
    write_json(root / "project.json", project)


def load(root: Path) -> dict:
    project = read_json(root / "project.json")
    if project.get("schema_version") != 1:
        raise ValueError("不支持此项目版本。")
    return project


def create(book: Path, reference: Path, root: Path, cfg: dict, encoding: str | None = None,
           rules_path: Path | None = None, reference_start: float = 0,
           heading_pattern: str | None = None) -> dict:
    if root.exists() and any(root.iterdir()):
        raise ValueError("输出目录不为空。新书请使用新目录；继续制作请使用 run。")
    text, detected = read_book(book, encoding)
    rules = read_json(rules_path) if rules_path else {}
    validate_pronunciations(rules)
    chapters = split_chapters(text, book.stem, heading_pattern)
    # Validate all text before touching the output directory.
    segments = []
    for chapter_index, chapter in enumerate(chapters, 1):
        chapter["id"] = f"c{chapter_index:04d}"
        for segment_index, content in enumerate(split_segments(chapter.pop("text"), cfg["max_chars"], rules), 1):
            spoken = apply_pronunciations(content, rules)
            segments.append({"id": f"{chapter['id']}_s{segment_index:04d}", "chapter_id": chapter["id"],
                             "text": content, "spoken_text": spoken, "text_hash": digest(spoken),
                             "kind": "pause" if is_structure(content) else "speech",
                             "state": "pause" if is_structure(content) else "pending", "attempts": 0, "total_attempts": 0,
                             "audio": None, "audio_sha256": None, "history": [], "check": None})
    if not any(segment["kind"] == "speech" for segment in segments):
        raise ValueError("书稿没有可朗读文字。场景分隔符会保留为停顿。")
    # Decode and validate the voice before creating a project. A bad input can be
    # retried at the same output path without removing a half-imported project.
    with tempfile.TemporaryDirectory(prefix="shuoshu-reference-") as temporary:
        prepared = Path(temporary) / "reference.wav"
        reference_info = prepare_reference(reference, prepared, cfg["ffmpeg"], reference_start)
        root.mkdir(parents=True, exist_ok=True)
        (root / "input").mkdir(exist_ok=True)
        shutil.copyfile(prepared, root / "input/reference.wav")
    inputs = root / "input"
    inputs.mkdir(exist_ok=True)
    shutil.copyfile(book, inputs / "source.txt")
    reference_copy = inputs / ("reference-original" + reference.suffix.lower())
    shutil.copyfile(reference, reference_copy)
    project = {"schema_version": 1, "title": book.stem, "created_at": now(), "encoding": detected,
               "source_sha256": file_hash(inputs / "source.txt"),
               "reference_sha256": file_hash(inputs / "reference.wav"), "reference_info": reference_info,
               "profile": generation_profile(cfg), "asset_signature": None,
               "book_config": book_settings(cfg), "qa_version": QA_VERSION,
               "chapter_pattern": heading_pattern,
               "preflight": {"chapters_confirmed": False, "voice_confirmed": False},
               "qa_profile": None, "segmentation_rules_hash": digest(rules),
               "chapters": chapters, "segments": segments, "exports": {}, "runs": []}
    write_json(root / "pronunciations.json", rules)
    save(root, project)
    return project


def invalidate(project: dict, segment: dict) -> None:
    previous = project["exports"].pop(segment["chapter_id"], None)
    if previous:
        project.setdefault("stale_exports", []).append(previous)


def _repair_boundaries(root: Path, project: dict, rules: dict) -> None:
    """Repair existing projects without recutting unaffected text or IDs."""
    source, _ = read_book(root / 'input/source.txt', project['encoding'])
    groups, cursor, previous_end, previous_chapter = [], 0, -1, None
    for segment in project['segments']:
        start = source.find(segment['text'], cursor)
        if start < 0:
            raise ValueError(f"片段 {segment['id']} 与保存的原稿不一致，无法修复分段边界。")
        if start != previous_end or segment['chapter_id'] != previous_chapter:
            groups.append([])
        groups[-1].append(segment)
        cursor = previous_end = start + len(segment['text'])
        previous_chapter = segment['chapter_id']
    revised = []
    for group in groups:
        original = [segment['text'] for segment in group]
        parts = align_segment_boundaries(original, rules)
        if parts == original:
            revised.extend(group)
            continue
        # Reuse unchanged segments first, including their audio and identity.
        by_text = defaultdict(deque)
        for index, segment in enumerate(group):
            by_text[segment['text']].append(index)
        indices = [by_text[part].popleft() if by_text[part] else None for part in parts]
        used = {index for index in indices if index is not None}
        remaining = iter(index for index in range(len(group)) if index not in used)
        for part, index in zip(parts, indices):
            if index is None:
                index = next(remaining)
                used.add(index)
            segment = group[index]
            if part != segment['text']:
                segment['history'].append({'stage': 'resegment', 'at': now(), 'text': segment['text']})
                segment['text'] = part
            revised.append(segment)
        for index, segment in enumerate(group):
            if index not in used:
                invalidate(project, segment)
                project.setdefault('retired_segments', []).append(segment)
    project['segments'] = revised


def refresh(root: Path, project: dict, cfg: dict) -> int:
    if project["profile"] != generation_profile(cfg):
        raise ValueError("声音模型、精度或语速已改变。为保持全书一致，请恢复配置或新建项目。")
    if inference_gap(cfg) != book_inference_gap(project):
        raise ValueError("片段内部生成停顿已改变。请恢复 inference_gap_ms 或新建项目，避免续跑时混用不同停顿。")
    if file_hash(root / "input/reference.wav") != project["reference_sha256"]:
        raise ValueError("参考录音已改变。请恢复原文件或新建项目，避免续跑时更换音色。")
    if file_hash(root / "input/source.txt") != project["source_sha256"]:
        raise ValueError("项目内原稿已改变。请用新稿新建项目，原项目的分段不会自动替换。")
    rules = read_json(root / "pronunciations.json")
    validate_pronunciations(rules)
    if project.get('segmentation_rules_hash') != digest(rules):
        _repair_boundaries(root, project, rules)
        project['segmentation_rules_hash'] = digest(rules)
    changed = 0
    for segment in project["segments"]:
        if is_structure(segment["text"]):
            if segment["state"] != "pause":
                segment["ever_required_review"] = bool(segment.get("ever_required_review") or segment["state"] in {"review", "error"})
                segment["history"].append({"stage": "structure", "at": now(), "previous_audio": segment.get("audio")})
                segment.update(kind="pause", state="pause", audio=None, audio_sha256=None, check=None)
                invalidate(project, segment)
                changed += 1
            continue
        segment["kind"] = "speech"
        if segment["state"] == "synthesizing":
            # An interrupted attempt did not finish. Keep its unique attempt number.
            segment["attempts"] = max(0, segment["attempts"] - 1)
            segment["state"] = "pending"
        elif segment["state"] == "checking":
            segment["state"] = "generated"
        spoken = apply_pronunciations(segment["text"], rules)
        if digest(spoken) != segment["text_hash"]:
            segment.update(spoken_text=spoken, text_hash=digest(spoken), state="pending", attempts=0,
                           audio=None, audio_sha256=None, check=None)
            invalidate(project, segment)
            changed += 1
        if segment["state"] in ACCEPTED | {"generated", "review"} and segment["audio"]:
            audio = contained(root, segment["audio"])
            if not audio.is_file() or file_hash(audio) != segment["audio_sha256"]:
                segment.update(state="pending", attempts=0, audio=None, audio_sha256=None, check=None)
                invalidate(project, segment)
                changed += 1
    if project.get("qa_profile") != cfg["qa"] or project.get("qa_version") != QA_VERSION:
        for segment in project["segments"]:
            if segment["state"] == "passed":
                segment["state"] = "generated"
                invalidate(project, segment)
        project["qa_profile"] = cfg["qa"].copy()
        project["qa_version"] = QA_VERSION
    project["book_config"] = book_settings(cfg)
    save(root, project)
    return changed


def select(project: dict, ids: list[str]) -> list[dict]:
    choices = {segment["id"]: segment for segment in project["segments"]}
    missing = set(ids) - set(choices)
    if missing:
        raise ValueError(f"找不到片段: {', '.join(sorted(missing))}")
    return [choices[item] for item in dict.fromkeys(ids)]


def retry(root: Path, ids: list[str]) -> None:
    project = load(root)
    chosen = select(project, ids)
    if any(is_structure(segment["text"]) for segment in chosen):
        raise ValueError("场景分隔符作为停顿导出，无需重新合成。")
    for segment in chosen:
        if segment["state"] in {"review", "error"}:
            segment["ever_required_review"] = True
        segment.setdefault("manual_actions", []).append({"at": now(), "action": "retry"})
        segment.update(state="pending", attempts=0, check=None)
        invalidate(project, segment)
    save(root, project)


def approve(root: Path, ids: list[str], note: str) -> None:
    project = load(root)
    chosen = select(project, ids)
    for segment in chosen:
        if segment["state"] not in {"review", "passed"} or not segment["audio"]:
            raise ValueError(f"{segment['id']} 尚无可验收的已校对音频。")
        audio = contained(root, segment["audio"])
        if file_hash(audio) != segment["audio_sha256"]:
            raise ValueError(f"{segment['id']} 音频已改变，请重新生成。")
        wav_info(audio)
    for segment in chosen:
        if segment["state"] == "review":
            segment["ever_required_review"] = True
        segment.setdefault("manual_actions", []).append({"at": now(), "action": "approve", "note": note})
        segment.update(state="approved", approval={"at": now(), "note": note})
    save(root, project)
