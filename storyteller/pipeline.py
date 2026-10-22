from __future__ import annotations

import time
from pathlib import Path

from .audio import export_chapter, wav_info
from .config import ROOT, asset_signature
from .project import ACCEPTED, invalidate, load, refresh, save, select
from .quality import evaluate
from .report import write_report
from .runtime import StageRunner, prerequisites
from .storage import FileLock, contained, digest, file_hash, now


def group_segments(project: dict) -> dict[str, list[dict]]:
    grouped = {}
    for segment in project["segments"]:
        grouped.setdefault(segment["chapter_id"], []).append(segment)
    return grouped


def export_ready(root: Path, project: dict, cfg: dict, chapters=None, grouped=None) -> int:
    grouped = group_segments(project) if grouped is None else grouped
    exported = 0
    for chapter in project["chapters"] if chapters is None else chapters:
        segments = grouped.get(chapter["id"], [])
        if not segments or any(s["state"] not in ACCEPTED for s in segments):
            project["exports"].pop(chapter["id"], None)
            continue
        if all(s["state"] == "pause" for s in segments):
            continue
        signature = digest({"audio": [cfg["scene_gap_ms"] if s["state"] == "pause" else s["audio_sha256"] for s in segments],
                            "gap": cfg["gap_ms"], "export": cfg["export"]})
        previous = project["exports"].get(chapter["id"])
        if previous and previous.get("signature") == signature:
            hashes = previous.get("sha256", {})
            if all(hashes.get(key) and contained(root / "exports", previous[key]).is_file()
                   and file_hash(contained(root / "exports", previous[key])) == hashes[key]
                   for key in ("wav", "mp3")):
                continue
        print(f"导出 {chapter['title']}", flush=True)
        if previous:
            invalidate(project, segments[0])
            save(root, project)
        result = export_chapter([int(cfg["scene_gap_ms"]) if s["state"] == "pause" else contained(root, s["audio"]) for s in segments], root / "exports",
                                chapter["id"], chapter["title"], cfg)
        hashes = {key: file_hash(contained(root / "exports", result[key])) for key in ("wav", "mp3")}
        project["exports"][chapter["id"]] = {**result, "signature": signature, "sha256": hashes}
        save(root, project)
        exported += 1
    return exported


def run(root: Path, cfg: dict, runner=None, check_assets: bool = True,
        segment_ids: list[str] | None = None) -> dict:
    root = root.resolve()
    if not (root / "project.json").is_file():
        raise ValueError(f"找不到项目文件: {root / 'project.json'}")
    if check_assets:
        missing = prerequisites(cfg)
        if missing:
            raise RuntimeError("运行环境尚未准备完成：\n" + "\n".join(missing) + "\n请运行 setup.ps1 -Models。")
    with FileLock(root / ".project.lock"), FileLock(ROOT / ".runtime/gpu.lock"):
        project = load(root)
        refresh(root, project, cfg)
        if check_assets:
            signature = asset_signature(cfg)
            if project["asset_signature"] and project["asset_signature"] != signature:
                raise RuntimeError("模型文件或推理源码已改变。请恢复模型版本或新建项目，避免音色漂移。")
            project["asset_signature"] = signature
        runner = runner or StageRunner(cfg)
        grouped = group_segments(project)
        indexed = {segment["id"]: segment for segment in project["segments"]}
        selected = {s["id"] for s in select(project, segment_ids)} if segment_ids is not None else None
        started = time.perf_counter()
        record = {"started_at": now(), "status": "running", "elapsed_seconds": 0,
                  "mode": "preview" if selected is not None else "production"}
        project["runs"].append(record)
        save(root, project)
        try:
            for chapter in project["chapters"]:
                segments = [s for s in grouped.get(chapter["id"], []) if selected is None or s["id"] in selected]
                if not segments:
                    continue
                chapter_changed = False
                while True:
                    if hasattr(runner, "check_cancelled"):
                        runner.check_cancelled()
                    pending = [s for s in segments if s["state"] == "pending" and s["attempts"] < 3]
                    if pending:
                        chapter_changed = True
                        print(f"{chapter['title']}：合成 {len(pending)} 段", flush=True)
                        jobs = []
                        for segment in pending:
                            attempt = segment["total_attempts"] + 1
                            relative = f"audio/{segment['id']}_a{attempt:03d}.wav"
                            jobs.append({"id": segment["id"], "text": segment["spoken_text"],
                                         "output": str(root / relative), "relative": relative,
                                         "seed": int(digest([segment["id"], attempt])[:8], 16)})
                        _stage(root, project, cfg, runner, "synth", jobs, indexed)
                    generated = [s for s in segments if s["state"] == "generated"]
                    if generated:
                        chapter_changed = True
                        print(f"{chapter['title']}：校对 {len(generated)} 段", flush=True)
                        jobs = [{"id": s["id"], "audio": str(contained(root, s["audio"]))} for s in generated]
                        _stage(root, project, cfg, runner, "check", jobs, indexed)
                    if not any(s["state"] in {"pending", "generated"} for s in segments):
                        break
                exported = export_ready(root, project, cfg, [chapter], grouped) if selected is None else 0
                if chapter_changed or exported:
                    write_report(root, project)
            record["status"] = "needs_review" if any(s["state"] in {"review", "error"} for s in project["segments"]) else "complete"
        except BaseException as exc:
            record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            record["error"] = str(exc)
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            record["finished_at"] = now()
            save(root, project)
            summary = write_report(root, project)
        return summary


def _stage(root, project, cfg, runner, stage, jobs, indexed=None):
    indexed = {s["id"]: s for s in project["segments"]} if indexed is None else indexed
    requests = {job["id"]: job for job in jobs}
    finished = set()
    for event in runner.events(stage, root, jobs):
        if event["event"] == "ready":
            print("模型已加载，开始处理。", flush=True)
            continue
        if event.get("id") not in requests:
            raise RuntimeError("工作进程返回了未知片段编号。")
        segment = indexed[event["id"]]
        if event["event"] == "started":
            if stage == "synth":
                segment["attempts"] += 1
                segment["total_attempts"] += 1
                segment.update(state="synthesizing", audio=requests[segment["id"]]["relative"], audio_sha256=None, check=None)
                invalidate(project, segment)
            else:
                segment["state"] = "checking"
        elif event["event"] in {"done", "error"}:
            finished.add(segment["id"])
            history = {"stage": stage, "at": now(), "attempt": segment["total_attempts"], **event}
            if stage == "synth":
                history["audio"] = segment["audio"]
            if event["event"] == "error":
                segment["error"] = event["error"]
                if stage == "check":
                    segment["state"] = "generated"
                    segment["history"].append(history)
                    save(root, project)
                    raise RuntimeError(f"校对服务失败，保留已生成音频：{event['error']}")
                segment["state"] = "pending" if segment["attempts"] < 3 else "error"
                segment.update(audio=None, audio_sha256=None)
            elif stage == "synth":
                try:
                    info = wav_info(contained(root, segment["audio"]))
                    if info["duration"] <= 0.2:
                        raise ValueError("生成音频过短")
                    segment["audio_sha256"] = file_hash(contained(root, segment["audio"]))
                    history["audio_sha256"] = segment["audio_sha256"]
                    segment["state"] = "generated"
                    segment.pop("error", None)
                except (ValueError, OSError, EOFError) as exc:
                    segment.update(state="pending" if segment["attempts"] < 3 else "error", error=str(exc), audio=None, audio_sha256=None)
            else:
                info = wav_info(contained(root, segment["audio"]))
                check = evaluate(segment["text"], event["recognized"], info["duration"], cfg["qa"], info["rms_db"])
                segment["check"] = check
                history["check"] = check
                segment["state"] = "passed" if check["passed"] else ("pending" if segment["attempts"] < 3 else "review")
                segment.pop("error", None)
            segment["history"].append(history)
            if segment["state"] in {"review", "error"}:
                segment["ever_required_review"] = True
            print(f"  {segment['id']} → {segment['state']}（本轮第 {segment['attempts']} 次）", flush=True)
        else:
            raise RuntimeError(f"无效的工作进程事件: {event['event']}")
        save(root, project)
    if finished != set(requests):
        raise RuntimeError("工作进程没有完成全部片段。已保存进度，请检查日志后续跑。")
