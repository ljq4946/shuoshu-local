from __future__ import annotations

import array
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave

from storyteller.config import DEFAULTS, ROOT, load_config
from storyteller.project import create, load


def tone(path: Path, seconds=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 24000
    samples = array.array("h", (int(4000 * math.sin(i * 2 * math.pi * 220 / rate)) for i in range(int(rate * seconds))))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


class BookTestCase(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".test-output"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="review-fixes-", dir=parent)
        self.base = Path(self.temporary.name).resolve()
        self.assertTrue(self.base.is_relative_to(parent.resolve()))
        self.addCleanup(self.cleanup_fixture)
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["ffmpeg"] = load_config()["ffmpeg"]
        if not self.cfg["ffmpeg"]:
            self.skipTest("The local FFmpeg runtime is required")

    def cleanup_fixture(self):
        if not self.base.is_relative_to((ROOT / ".test-output").resolve()):
            raise RuntimeError("Test cleanup path escapes the workspace")
        self.temporary.cleanup()

    def book(self, name="book", text="这是一段用于验证本地制作及断点续跑的工程测试文字。", cfg=None):
        source, reference = self.base / (name + ".txt"), self.base / (name + ".wav")
        source.write_text(text, encoding="utf-8")
        tone(reference)
        root = self.base / name
        create(source, reference, root, cfg or self.cfg)
        return root


class Runner:
    def __init__(self, *, forbid_synth=False, recognized=None, fail_check=False):
        self.forbid_synth, self.recognized, self.fail_check = forbid_synth, recognized, fail_check
        self.calls = []

    def events(self, stage, root, jobs):
        self.calls.append((stage, [job["id"] for job in jobs]))
        if self.forbid_synth and stage == "synth":
            raise AssertionError("Existing audio should have been reused")
        texts = {s["id"]: s["text"] for s in load(root)["segments"]}
        yield {"event": "ready"}
        for job in jobs:
            yield {"event": "started", "id": job["id"]}
            if stage == "synth":
                tone(Path(job["output"]), max(2, len(texts[job["id"]]) / 4))
                yield {"event": "done", "id": job["id"]}
            elif self.fail_check:
                yield {"event": "error", "id": job["id"], "error": "controlled ASR failure"}
            else:
                yield {"event": "done", "id": job["id"], "recognized": self.recognized or texts[job["id"]]}
