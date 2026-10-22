"""Book preparation and explicit human checkpoints, independent of the UI."""
from __future__ import annotations

import copy
import shutil
import uuid
from pathlib import Path

from .audio import concatenate, prepare_reference, wav_info
from .pipeline import run
from .project import load, save
from .storage import FileLock, contained, digest, file_hash, now, read_json, write_json
from .text import apply_pronunciations, is_structure, read_book, split_chapters, split_segments, validate_pronunciations


def require_confirmation(root: Path) -> None:
    gate = load(root).get("preflight")
    # Existing books may already be in production; preserve their resume path.
    if gate is None:
        return
    missing = []
    if not gate.get("chapters_confirmed"):
        missing.append("章节未确认（confirm --chapters）")
    if not gate.get("voice_confirmed"):
        missing.append("试读音色未确认（先 preview，再 confirm --voice）")
    if missing:
        raise ValueError("；".join(missing) + "。也可使用 studio 在本地面板完成。")


def confirm(root: Path, *, chapters=False, voice=False, note="已检查并确认") -> dict:
    if not chapters and not voice:
        raise ValueError("请选择确认章节或试读音色。")
    with FileLock(root / ".project.lock"):
        project = load(root)
        gate = project.setdefault("preflight", {})
        if voice:
            sample = gate.get("preview")
            if not sample or not contained(root, sample["audio"]).is_file():
                raise ValueError("请先生成并试听试读音频。")
            if file_hash(contained(root, sample["audio"])) != sample["sha256"]:
                raise ValueError("试读音频已经改变，请重新生成试读。")
            if sample["profile"] != project["profile"] or sample["reference_sha256"] != project["reference_sha256"]:
                raise ValueError("试读声音与当前配置不一致，请重新试读。")
            gate.update(voice_confirmed=True, voice_confirmed_at=now(), voice_note=note)
        if chapters:
            gate.update(chapters_confirmed=True, chapters_confirmed_at=now(), chapters_note=note)
        save(root, project)
        return gate


def preview(root: Path, cfg: dict, runner=None, check_assets=True) -> dict:
    project = load(root)
    ids, length = [], 0
    for segment in project["segments"]:
        if is_structure(segment["text"]):
            continue
        ids.append(segment["id"])
        length += len(segment["text"])
        if length >= 240 or len(ids) >= 4:
            break
    if not ids:
        raise ValueError("没有可试读的文字。")
    summary = run(root, cfg, runner, check_assets, segment_ids=ids)
    with FileLock(root / ".project.lock"):
        project = load(root)
        selected = [s for s in project["segments"] if s["id"] in ids]
        if any(not s.get("audio") for s in selected):
            raise RuntimeError("试读生成失败，请查看异常片段和日志后重试。")
        paths = [contained(root, s["audio"]) for s in selected]
        for segment, path in zip(selected, paths):
            if file_hash(path) != segment["audio_sha256"]:
                raise ValueError("试读片段已改变，请重试。")
        folder = root / "preview"
        folder.mkdir(exist_ok=True)
        temporary = folder / "voice.tmp.wav"
        concatenate(paths, temporary, cfg["gap_ms"])
        temporary.replace(folder / "voice.wav")
        sample = {"audio": "preview/voice.wav", "sha256": file_hash(folder / "voice.wav"),
                  "duration": wav_info(folder / "voice.wav")["duration"], "segments": ids,
                  "profile": project["profile"], "reference_sha256": project["reference_sha256"], "at": now()}
        project.setdefault("preflight", {})["preview"] = sample
        save(root, project)
    return summary


def replan(root: Path, cfg: dict, *, pattern: str | None = None, starts: list[int] | None = None) -> dict:
    with FileLock(root / ".project.lock"):
        project = load(root)
        if project["exports"] or any(record.get("mode") != "preview" for record in project["runs"]):
            raise ValueError("已开始全书制作，章节规划已固定。请用原稿创建新项目重新划分。")
        if file_hash(root / "input/source.txt") != project["source_sha256"]:
            raise ValueError("项目原稿已改变，请恢复原稿。")
        text, _ = read_book(root / "input/source.txt", project["encoding"])
        chapters = split_chapters(text, project["title"], pattern, starts)
        rules = read_json(root / "pronunciations.json")
        validate_pronunciations(rules)
        generated = any(s["total_attempts"] for s in project["segments"])
        suffix = "_r" + uuid.uuid4().hex[:8] if generated else ""
        segments = []
        for chapter_index, chapter in enumerate(chapters, 1):
            chapter["id"] = f"c{chapter_index:04d}"
            for segment_index, content in enumerate(split_segments(chapter.pop("text"), cfg["max_chars"], rules), 1):
                spoken = apply_pronunciations(content, rules)
                pause = is_structure(content)
                segments.append({"id": f"{chapter['id']}_s{segment_index:04d}{suffix}", "chapter_id": chapter["id"],
                                 "text": content, "spoken_text": spoken, "text_hash": digest(spoken),
                                 "kind": "pause" if pause else "speech", "state": "pause" if pause else "pending",
                                 "attempts": 0, "total_attempts": 0, "audio": None, "audio_sha256": None,
                                 "check": None, "history": []})
        if generated:
            project.setdefault("retired_segments", []).extend(project["segments"])
        project.update(chapters=chapters, segments=segments, chapter_pattern=pattern,
                       chapter_starts=sorted(set(starts)) if starts is not None else None,
                       segmentation_rules_hash=digest(rules))
        gate = project.setdefault("preflight", {})
        gate.update(chapters_confirmed=False, voice_confirmed=False)
        gate.pop("preview", None)
        save(root, project)
        return project


def recut_reference(root: Path, cfg: dict, start: float) -> dict:
    """Allow auditioning a different crop before production, keeping the original."""
    with FileLock(root / ".project.lock"):
        project = load(root)
        if project["exports"] or any(r.get("mode") != "preview" for r in project["runs"]):
            raise ValueError("已开始全书制作，参考声音已固定。更换录音请新建项目。")
        sources = list((root / "input").glob("reference-original.*"))
        if len(sources) != 1:
            raise ValueError("找不到唯一的原始参考录音。")
        candidate = root / "input/reference-candidate.wav"
        try:
            info = prepare_reference(sources[0], candidate, cfg["ffmpeg"], start)
            candidate.replace(root / "input/reference.wav")
        finally:
            candidate.unlink(missing_ok=True)
        project.update(reference_sha256=file_hash(root / "input/reference.wav"), reference_info=info)
        for segment in project["segments"]:
            if segment["state"] == "pause":
                continue
            segment["history"].append({"stage": "reference-recut", "at": now(), "previous_audio": segment["audio"]})
            segment.update(state="pending", attempts=0, audio=None, audio_sha256=None, check=None)
        project["preflight"] = {"chapters_confirmed": project.get("preflight", {}).get("chapters_confirmed", False),
                                "voice_confirmed": False}
        save(root, project)
        return info


def update_pronunciations(root: Path, cfg: dict, rules: dict) -> dict:
    from .project import refresh
    validate_pronunciations(rules)
    with FileLock(root / ".project.lock"):
        project = load(root)
        # Validate immutable inputs before replacing the dictionary.
        refresh(root, project, cfg)
        previous = read_json(root / "pronunciations.json")
        try:
            write_json(root / "pronunciations.json", rules)
            changed = refresh(root, project, cfg)
        except Exception:
            write_json(root / "pronunciations.json", previous)
            raise
        return {"changed_segments": changed}


def update_settings(root: Path, cfg: dict, settings: dict) -> dict:
    from .config import book_inference_gap, book_settings, generation_profile, inference_gap, validate_config
    permitted = {"speed", "gap_ms", "scene_gap_ms", "mp3_bitrate"}
    if not isinstance(settings, dict) or set(settings) - permitted:
        raise ValueError("无效的制作设置。")
    candidate = copy.deepcopy(cfg)
    # The UI edits only chapter joins. Freeze the synthesis gap before applying
    # the new join value, including callers holding a pre-import default config.
    candidate["inference_gap_ms"] = inference_gap(cfg)
    for key, value in settings.items():
        if key == "mp3_bitrate":
            if value not in {"64k", "96k", "128k", "192k", "256k", "320k"}:
                raise ValueError("请选择支持的 MP3 码率。")
            candidate["export"]["mp3_bitrate"] = value
        else:
            candidate[key] = value
    validate_config(candidate)
    with FileLock(root / ".project.lock"):
        project = load(root)
        if candidate["inference_gap_ms"] != book_inference_gap(project):
            raise ValueError("片段内部生成停顿已改变。请恢复 inference_gap_ms 或新建项目，避免续跑时混用不同停顿。")
        if candidate["speed"] != project["profile"]["speed"]:
            if project["exports"] or any(r.get("mode") != "preview" for r in project["runs"]):
                raise ValueError("已开始全书制作，语速已固定。调整语速请新建项目。")
            for segment in project["segments"]:
                if segment["state"] == "pause":
                    continue
                segment["history"].append({"stage": "speed-change", "at": now(), "previous_audio": segment["audio"]})
                segment.update(state="pending", attempts=0, audio=None, audio_sha256=None, check=None)
            project["profile"] = generation_profile(candidate)
            project.setdefault("preflight", {}).update(voice_confirmed=False)
            project["preflight"].pop("preview", None)
        if any(candidate[key] != cfg[key] for key in ("gap_ms", "scene_gap_ms", "export")):
            project.setdefault("stale_exports", []).extend(project["exports"].values())
            project["exports"] = {}
        project["book_config"] = book_settings(candidate)
        save(root, project)
    return candidate


def resource_estimate(root: Path, project: dict, cfg: dict) -> dict:
    chars = sum(len(s["text"]) for s in project["segments"] if not is_structure(s["text"]))
    seconds = chars / 4
    # Three attempts, WAV master, normalized/concatenated temporary WAVs + MP3.
    estimated = int(seconds * (24000 * 2 * 6 + 16000) * 1.25)
    free = shutil.disk_usage(root).free
    return {"estimated_audio_minutes": round(seconds / 60, 1), "estimated_working_bytes": estimated,
            "free_bytes": free, "disk_sufficient": free >= estimated,
            "basis": "按每秒 4 字、24 kHz PCM16、最多三次尝试及导出临时文件估算；实际可能不同"}


LISTENING_ITEMS = ("target_voice", "voice_similarity", "voice_consistency", "sentence_integrity", "pacing", "joins")


def record_listening(root: Path, minutes: float, checks: dict, note: str) -> dict:
    if not isinstance(minutes, (int, float)) or not 0 <= minutes <= 100000:
        raise ValueError("试听分钟数无效。")
    if not isinstance(checks, dict) or set(checks) != set(LISTENING_ITEMS) or any(type(v) is not bool for v in checks.values()):
        raise ValueError("请分别填写完整的试听检查项。")
    with FileLock(root / ".project.lock"):
        project = load(root)
        available = sum(output.get("duration", 0) for output in project["exports"].values()) / 60
        if minutes > available + 0.01:
            raise ValueError("试听时长不能超过当前已导出的音频总长。")
        passed = minutes >= 10 and available >= 10 and all(checks.values())
        result = {"at": now(), "minutes": minutes, "available_minutes": round(available, 3),
                  "checks": checks, "note": str(note), "accepted": passed,
                  "status": "human_accepted" if passed else "pending_human_acceptance",
                  "reference_sha256": project["reference_sha256"], "profile": project["profile"],
                  "exports": copy.deepcopy(project["exports"])}
        write_json(root / "listening.json", result)
        return result
