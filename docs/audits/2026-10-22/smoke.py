"""Real offline TTS/ASR and chapter export in an isolated audit project.

Does not assert human listening acceptance or modify an existing book.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from storyteller.config import load_book_config
from storyteller.pipeline import export_ready
from storyteller.preparation import preview, require_confirmation
from storyteller.project import create, load
from storyteller.storage import FileLock, file_hash, now, write_json


def main():
    output = Path(__file__).resolve().parent
    original = ROOT / "books/十年非梦_春日野穹"
    cfg = load_book_config(original)
    source = output / "smoke-source.txt"
    source.write_text("夜色渐深，窗外的风慢慢停了下来。他合上书，轻轻关上了台灯。", encoding="utf-8")
    root = output / "model-smoke"
    create(source, original / "input/reference.wav", root, cfg)
    preview(root, cfg)
    project = load(root)
    confirmation_blocks_production = False
    try:
        require_confirmation(root)
    except ValueError:
        confirmation_blocks_production = True
    with FileLock(root / ".project.lock"):
        exported = export_ready(root, project, cfg)
    result = {"checked_at_utc": now(), "profile": project["profile"],
              "preflight": {key: project["preflight"].get(key) for key in ("chapters_confirmed", "voice_confirmed")},
              "full_production_blocked_without_human_confirmation": confirmation_blocks_production,
              "scope": "Real preview plus separate export of automatically accepted segments; no human confirmation recorded",
              "segments": [{key: s.get(key) for key in ("id", "text", "state", "check", "audio_sha256")} for s in project["segments"]],
              "exports_created": exported, "exports": project["exports"],
              "preview_sha256": file_hash(root / "preview/voice.wav"),
              "human_listening_performed": False}
    write_json(output / "model-smoke.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
