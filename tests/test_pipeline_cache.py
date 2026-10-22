from __future__ import annotations

import contextlib
import io

from storyteller.pipeline import run
from storyteller.project import load, save
from storyteller.quality import QA_VERSION
from storyteller.storage import file_hash
from tests.support import BookTestCase, Runner


class QualityUpgradeTests(BookTestCase):
    def test_old_numeric_false_pass_is_rechecked_without_resynthesis(self):
        text = "行程约3-5公里，请沿着河边的小路继续向前走。"
        root = self.book(text=text)
        with contextlib.redirect_stdout(io.StringIO()):
            run(root, self.cfg, Runner(), False)
        project = load(root)
        segment = project["segments"][0]
        audio_hash = segment["audio_sha256"]
        project["qa_version"] = 2
        segment.update(attempts=3, total_attempts=3)
        save(root, project)
        runner = Runner(forbid_synth=True, recognized=text.replace("3-5", "35"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(root, self.cfg, runner, False)
        after = load(root)
        self.assertEqual(after["qa_version"], QA_VERSION)
        self.assertEqual([stage for stage, _ in runner.calls], ["check"])
        self.assertEqual(after["segments"][0]["audio_sha256"], audio_hash)
        self.assertEqual(after["segments"][0]["state"], "review")
        self.assertIn("关键数值或正负号不一致", after["segments"][0]["check"]["reasons"])
        self.assertEqual(result["ready_chapters"], 0)
        self.assertEqual(len(list((root / "exports/history").glob("*"))), 2)


class ResumeAndExportTests(BookTestCase):
    def test_equal_size_export_corruption_is_repaired_without_resynthesis(self):
        root = self.book()
        with contextlib.redirect_stdout(io.StringIO()):
            run(root, self.cfg, Runner(), False)
        before = load(root)
        output = root / "exports" / before["exports"]["c0001"]["mp3"]
        data = bytearray(output.read_bytes())
        data[-1] ^= 1
        output.write_bytes(data)
        runner = Runner(forbid_synth=True)
        with contextlib.redirect_stdout(io.StringIO()):
            run(root, self.cfg, runner, False)
        after = load(root)
        self.assertEqual(runner.calls, [])
        self.assertEqual(before["segments"][0]["audio_sha256"], after["segments"][0]["audio_sha256"])
        self.assertEqual(file_hash(output), after["exports"]["c0001"]["sha256"]["mp3"])
        self.assertEqual(len(list((root / "exports/history").glob("*"))), 2)

    def test_asr_failure_resumes_from_generated_audio(self):
        root = self.book()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "controlled ASR failure"):
            run(root, self.cfg, Runner(fail_check=True), False)
        before = load(root)
        self.assertEqual(before["segments"][0]["state"], "generated")
        runner = Runner(forbid_synth=True)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = run(root, self.cfg, runner, False)
        after = load(root)
        self.assertEqual([stage for stage, _ in runner.calls], ["check"])
        self.assertEqual(before["segments"][0]["audio_sha256"], after["segments"][0]["audio_sha256"])
        self.assertEqual(after["segments"][0]["state"], "passed")
        self.assertEqual(summary["ready_chapters"], 1)
