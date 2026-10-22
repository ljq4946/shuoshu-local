"""Real offline smoke check after the five fixes, isolated from real books."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from storyteller.config import load_book_config
from storyteller.pipeline import export_ready, run
from storyteller.preparation import preview, require_confirmation, update_settings
from storyteller.project import create, load
from storyteller.storage import FileLock, file_hash, now, write_json


def main():
    output = Path(__file__).resolve().parent
    original = ROOT / "books/十年非梦_春日野穹"
    cfg = load_book_config(original)
    source = output / "smoke-source.txt"
    source.write_text("夜色渐深，窗外的风慢慢停了下来。他合上书，轻轻关上了台灯。\n"
                      "现在温度零下五度，明天十二点三十分出发，步行三到五公里，再取总数的二分之一。", encoding="utf-8")
    root = output / "model-smoke"
    create(source, original / "input/reference.wav", root, cfg)
    preview(root, cfg)
    before = load(root)
    old_hashes = [s["audio_sha256"] for s in before["segments"]]
    changed = update_settings(root, cfg, {"gap_ms": 1000})
    run(root, changed, segment_ids=[s["id"] for s in before["segments"]])
    project = load(root)
    assert old_hashes == [s["audio_sha256"] for s in project["segments"]]
    blocked = False
    try:
        require_confirmation(root)
    except ValueError:
        blocked = True
    with FileLock(root / ".project.lock"):
        exported = export_ready(root, project, changed)
    job_files = list((root / ".jobs").glob("synth-*.json"))
    requests = [json.loads(path.read_text(encoding="utf-8")) for path in job_files]
    passed = all(s["state"] == "passed" for s in project["segments"])
    result = {"checked_at_utc": now(), "passed": passed and exported == 1 and blocked,
              "human_listening_performed": False, "profile": project["profile"],
              "inference_gap_ms": changed["inference_gap_ms"], "export_gap_ms": changed["gap_ms"],
              "all_cached_audio_reused_after_gap_change": True,
              "synthesis_stage_requests": len(job_files),
              "requested_internal_gaps_ms": [r["config"]["inference_gap_ms"] for r in requests],
              "preflight": {key: project["preflight"].get(key) for key in ("chapters_confirmed", "voice_confirmed")},
              "full_production_blocked_without_human_confirmation": blocked,
              "segments": [{key: s.get(key) for key in ("id", "text", "state", "check", "audio_sha256")} for s in project["segments"]],
              "exports": project["exports"], "preview_sha256": file_hash(root / "preview/voice.wav")}
    write_json(output / "model-smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
