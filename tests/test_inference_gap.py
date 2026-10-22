from __future__ import annotations

import contextlib
import copy
import io

from storyteller.config import DEFAULTS, generation_profile, load_book_config, validate_config
from storyteller.pipeline import run
from storyteller.preparation import confirm, preview, update_settings
from storyteller.project import load, refresh, save
from storyteller.storage import file_hash, write_json
from storyteller.worker import synthesize
from tests.support import BookTestCase, Runner


class Recorder:
    def __init__(self):
        self.options = []

    def infer_generator(self, **options):
        self.options.append(options)
        yield None

    def infer(self, **options):
        self.options.append(options)


class InferenceGapTests(BookTestCase):
    def test_export_gap_changes_leave_synthesis_and_preview_unchanged(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(inference_mode="sentence", gap_ms=140)
        root = self.book(cfg=cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            preview(root, cfg, Runner(), False)
        confirm(root, chapters=True, voice=True, note="test fixture")
        before = load(root)
        cfg = load_book_config(root)
        changed = update_settings(root, cfg, {"gap_ms": 1000})
        self.assertEqual(changed["inference_gap_ms"], 140)
        self.assertEqual(generation_profile(changed), generation_profile(cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            runner = Runner(forbid_synth=True)
            run(root, changed, runner, False)
        after = load(root)
        self.assertEqual(runner.calls, [])
        self.assertEqual(before["segments"][0]["audio_sha256"], after["segments"][0]["audio_sha256"])
        self.assertEqual(before["preflight"], after["preflight"])
        reloaded = load_book_config(root)
        self.assertEqual((reloaded["gap_ms"], reloaded["inference_gap_ms"]), (1000, 140))
        model = Recorder()
        for settings in (cfg, changed, reloaded):
            synthesize(model, settings, root / "input/reference.wav", "旧片段和重做片段保持相同内部停顿。", self.base / "unused.wav")
        self.assertEqual([x["interval_silence"] for x in model.options], [140, 140, 140])

    def test_old_book_restores_its_original_internal_gap_before_config_overrides(self):
        for mode, expected in (("sentence", 140), ("legacy", 200)):
            with self.subTest(mode=mode):
                cfg = copy.deepcopy(self.cfg)
                cfg.update(inference_mode=mode, gap_ms=140)
                root = self.book(name=mode, cfg=cfg)
                project = load(root)
                project["book_config"].pop("inference_gap_ms")
                save(root, project)
                config = self.base / (mode + "-config.json")
                write_json(config, {"gap_ms": 1000, "inference_gap_ms": None})
                loaded = load_book_config(root, config)
                self.assertEqual(loaded["inference_gap_ms"], expected)
                self.assertEqual(loaded["gap_ms"], 1000)
                refresh(root, load(root), loaded)
                self.assertEqual(load(root)["book_config"]["inference_gap_ms"], expected)

    def test_internal_gap_change_is_rejected_without_modifying_existing_book(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(inference_mode="sentence", gap_ms=140, inference_gap_ms=140)
        root = self.book(cfg=cfg)
        original = file_hash(root / "project.json")
        changed = copy.deepcopy(cfg)
        changed["inference_gap_ms"] = 300
        with self.assertRaisesRegex(ValueError, "内部生成停顿已改变"):
            refresh(root, load(root), changed)
        with self.assertRaisesRegex(ValueError, "内部生成停顿已改变"):
            update_settings(root, changed, {"gap_ms": 1000})
        self.assertEqual(file_hash(root / "project.json"), original)

    def test_legacy_worker_preserves_its_default_200ms_internal_gap(self):
        model = Recorder()
        cfg = copy.deepcopy(DEFAULTS)
        cfg["gap_ms"] = 1000
        synthesize(model, cfg, self.base / "ref.wav", "工程样例", self.base / "unused.wav")
        self.assertEqual(model.options[0]["interval_silence"], 200)

    def test_invalid_internal_gap_is_rejected(self):
        for gap in (-1, 3001, True, 1.5):
            with self.subTest(gap=gap):
                cfg = copy.deepcopy(DEFAULTS)
                cfg["inference_gap_ms"] = gap
                with self.assertRaisesRegex(ValueError, "inference_gap_ms"):
                    validate_config(cfg)
