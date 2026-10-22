"""Isolated audit reproductions; does not modify source, real books or models.

Run from the project root with .runtime/core/Scripts/python.exe -B -X utf8
docs/audits/2026-10-22/reproduce.py. The assertions validate the fixtures;
`reproduced` describes an observed defect, not a passing product test.
"""
from __future__ import annotations

import array
import contextlib
import copy
import http.cookiejar
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from storyteller.config import generation_profile, load_config
from storyteller.pipeline import run
from storyteller.preparation import update_settings
from storyteller.project import create, load, save
from storyteller.quality import evaluate, normalize
from storyteller.storage import file_hash, write_json
from storyteller.studio import Studio, StudioServer
from storyteller.text import split_chapters
from storyteller.worker import synthesize


def tone(path: Path, seconds=5.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array.array("h", (int(4000 * math.sin(i * 2 * math.pi * 220 / 24000))
                                for i in range(int(24000 * seconds))))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


def make_book(base: Path, name: str, cfg: dict, text: str):
    source, reference = base / (name + ".txt"), base / (name + ".wav")
    source.write_text(text, encoding="utf-8")
    tone(reference)
    output = base / name
    create(source, reference, output, cfg)
    return output


class Runner:
    def __init__(self, fail_check=False, forbid_synth=False):
        self.calls = []
        self.fail_check, self.forbid_synth = fail_check, forbid_synth

    def events(self, stage, root, jobs):
        self.calls.append({"stage": stage, "jobs": [j["id"] for j in jobs]})
        if self.forbid_synth and stage == "synth":
            raise AssertionError("Unexpected resynthesis in fixture")
        texts = {s["id"]: s["text"] for s in load(root)["segments"]}
        yield {"event": "ready"}
        for job in jobs:
            yield {"event": "started", "id": job["id"]}
            if stage == "synth":
                tone(Path(job["output"]), max(2, len(texts[job["id"]]) / 4))
                yield {"event": "done", "id": job["id"]}
            elif self.fail_check:
                yield {"event": "error", "id": job["id"], "error": "isolated ASR failure"}
            else:
                yield {"event": "done", "id": job["id"], "recognized": texts[job["id"]]}


def numeric_checks(cfg):
    examples = [
        ("range", "行程约3-5公里，请沿着河边的小路一直向前走。", "行程约35公里，请沿着河边的小路一直向前走。"),
        ("fraction", "取总数的1/2，然后按顺序放入盒子里面保存。", "取总数的12，然后按顺序放入盒子里面保存。"),
        ("list", "编号1、2、3，请逐项完成并保存这次检查的全部记录。", "编号123，请逐项完成并保存这次检查的全部记录。"),
    ]
    wrong = [{"case": name, "expected": a, "recognized": b,
              "normalized_expected": normalize(a), "normalized_recognized": normalize(b),
              "result": evaluate(a, b, 8, cfg["qa"])} for name, a, b in examples]
    controls = [
        ("negative_sign", "预算-5元，请仔细核对数额并记录下来。", "预算5元，请仔细核对数额并记录下来。", False),
        ("negative_equivalence", "温度为-5度。", "温度为零下五度。", True),
        ("time_equivalence", "出发时间12:30。", "出发时间十二点三十分。", True),
        ("date_equivalence", "今天是2026/10/21。", "今天是二零二六年十月二十一日。", True),
    ]
    previous = [{"case": name, "expected_pass": passed,
                 "result": evaluate(a, b, 4, cfg["qa"])} for name, a, b, passed in controls]
    assert all(c["result"]["passed"] == c["expected_pass"] for c in previous)
    return {"reproduced": all(c["result"]["passed"] for c in wrong),
            "false_acceptances": wrong, "previous_regression_controls": previous}


def critical_checks(cfg):
    examples = [
        ("AI", "这次讲解围绕AI技术的研究展开，请各位认真核对每个环节并保存相关文字资料，以便后续继续阅读。", "AI", ""),
        ("一鸣", "这次讨论结束以后，一鸣沿着门口的小路返回住处，准备继续核对今天整理好的材料并保存记录。", "一鸣", "一明"),
    ]
    results = []
    for term, a, old, new in examples:
        qa = copy.deepcopy(cfg["qa"])
        qa["critical_terms"] = [term]
        b = a.replace(old, new)
        results.append({"critical_term": term, "expected": a, "recognized": b,
                        "result": evaluate(a, b, 12, qa)})
    return {"reproduced": all(r["result"]["passed"] for r in results), "cases": results}


def chapter_checks():
    template = "第一章 开始\n这是正文。\n{}\n后来发生了很多事情。\n第二章 结束\n这是结尾。"
    cases = []
    for line in ("第一回见面时我们都还年轻", "第一回见面时……", "第一回见面时我们都还年轻。"):
        titles = [c["title"] for c in split_chapters(template.format(line), "测试")]
        cases.append({"body_line": line, "expected_chapters": 2, "actual_titles": titles})
    assert len(cases[-1]["actual_titles"]) == 2
    return {"reproduced": all(len(c["actual_titles"]) == 3 for c in cases[:2]), "cases": cases}


def cache_checks(base, cfg):
    root = make_book(base, "cache", cfg, "这是一段用于核验断点续跑以及导出缓存的工程测试文字。")
    first = Runner()
    run(root, cfg, first, False)
    before = load(root)
    assert all(s["state"] == "passed" for s in before["segments"])
    output = root / "exports" / before["exports"]["c0001"]["mp3"]
    content = bytearray(output.read_bytes())
    content[-1] ^= 1
    output.write_bytes(content)
    resume = Runner(forbid_synth=True)
    result = run(root, cfg, resume, False)
    after = load(root)
    assert not resume.calls and result["ready_chapters"] == 1
    assert file_hash(output) == after["exports"]["c0001"]["sha256"]["mp3"]
    assert [s["audio_sha256"] for s in before["segments"]] == [s["audio_sha256"] for s in after["segments"]]
    return {"passed": True, "same_size_corruption_repaired": True, "resume_model_calls": resume.calls,
            "segment_audio_reused": True, "archived_files": len(list((root / "exports/history").glob("*")))}


def asr_failure_checks(base, cfg):
    root = make_book(base, "asr", cfg, "这是一段用于核验识别服务失败之后能否继续处理的工程测试文字。")
    failing = Runner(fail_check=True)
    try:
        run(root, cfg, failing, False)
    except RuntimeError as exc:
        assert "isolated ASR failure" in str(exc)
    else:
        raise AssertionError("ASR fixture did not fail")
    failed = load(root)
    assert all(s["state"] == "generated" and s["audio_sha256"] for s in failed["segments"])
    resume = Runner(forbid_synth=True)
    summary = run(root, cfg, resume, False)
    assert [c["stage"] for c in resume.calls] == ["check"]
    assert summary["ready_chapters"] == 1
    return {"passed": True, "after_error_states": [s["state"] for s in failed["segments"]],
            "resume_calls": resume.calls, "ready_chapters": summary["ready_chapters"]}


def gap_checks(base, config):
    cfg = copy.deepcopy(config)
    cfg.update(inference_mode="sentence", max_chars=500)
    root = make_book(base, "gap", cfg, "我们沿着河边慢慢向前走，听见远处传来钟声，继续观察身边的景物。" * 5)
    run(root, cfg, Runner(), False)
    before = load(root)
    changed = update_settings(root, cfg, {"gap_ms": 1000})
    resume = Runner(forbid_synth=True)
    run(root, changed, resume, False)
    after = load(root)

    class Recorder:
        def __init__(self):
            self.options = []

        def infer_generator(self, **options):
            self.options.append(options)
            yield None

    model = Recorder()
    for settings in (cfg, changed):
        synthesize(model, settings, root / "input/reference.wav", before["segments"][0]["text"], base / "unused.wav")
    reused = [s["audio_sha256"] for s in before["segments"]] == [s["audio_sha256"] for s in after["segments"]]
    return {"reproduced": reused and not resume.calls and generation_profile(cfg) == generation_profile(changed),
            "evidence_kind": "real controller/FFmpeg with simulated model; worker options recorded",
            "internal_inference_gap_before_ms": model.options[0]["interval_silence"],
            "internal_inference_gap_after_ms": model.options[1]["interval_silence"],
            "generation_profiles_equal": generation_profile(cfg) == generation_profile(changed),
            "cached_segment_audio_reused": reused, "resume_model_calls": resume.calls,
            "chapter_export_signature_changed": before["exports"]["c0001"]["signature"] != after["exports"]["c0001"]["signature"]}


def http_checks(base, cfg):
    roots = [make_book(base, "http" + str(i), cfg, "这是一段用于本地面板会话与音频访问检查的文字。") for i in range(2)]
    studios = [Studio(root) for root in roots]
    servers = [StudioServer(studio) for studio in studios]
    threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True) for s in servers]
    for thread in threads:
        thread.start()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def status(url, headers=None):
        request = urllib.request.Request(url, headers=headers or {})
        try:
            with opener.open(request, timeout=5) as response:
                response.read()
                return response.status
        except urllib.error.HTTPError as exc:
            exc.read()
            return exc.code

    try:
        assert status(servers[0].url) == 200
        media_a = status(servers[0].origin + "/media/input/reference.wav")
        assert media_a == 200
        assert status(servers[1].url) == 200
        media_b = status(servers[1].origin + "/media/input/reference.wav")
        overwritten_a = status(servers[0].origin + "/media/input/reference.wav")
        api_a = status(servers[0].origin + "/api/state", {"X-Shuoshu-Token": studios[0].token})
        range_b = status(servers[1].origin + "/media/input/reference.wav", {"Range": "bytes=0-31"})
        traversal_b = status(servers[1].origin + "/media/../project.json")
        anonymous = urllib.request.build_opener()
        try:
            anonymous.open(servers[1].origin + "/api/state", timeout=5)
        except urllib.error.HTTPError as exc:
            unauthenticated = exc.code
        else:
            unauthenticated = 200
        assert api_a == 200 and media_b == 200 and range_b == 206 and traversal_b == 404 and unauthenticated == 403
        return {"reproduced": overwritten_a == 403, "single_panel_audio_http_status": media_a,
                "second_panel_audio_http_status": media_b, "first_panel_after_second_session_http_status": overwritten_a,
                "first_panel_api_with_token_http_status": api_a, "range_request_http_status": range_b,
                "path_traversal_http_status": traversal_b, "anonymous_api_http_status": unauthenticated}
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)


def main():
    output = Path(__file__).resolve().parent
    cfg = load_config()
    results = {"numeric_boundaries": numeric_checks(cfg), "critical_terms": critical_checks(cfg),
               "chapter_detection": chapter_checks()}
    captured = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="audit-fixtures-", dir=output) as folder:
        base = Path(folder).resolve()
        assert base.is_relative_to(output) and output.is_relative_to(ROOT)
        with contextlib.redirect_stdout(captured):
            results["export_cache_regression"] = cache_checks(base, cfg)
            results["asr_resume_regression"] = asr_failure_checks(base, cfg)
            results["sentence_gap_cache"] = gap_checks(base, cfg)
            results["multi_panel_media_session"] = http_checks(base, cfg)
    write_json(output / "reproductions.json", results)
    (output / "reproductions.log").write_text(captured.getvalue(), encoding="utf-8")
    print(json.dumps({key: {field: value[field] for field in ("reproduced", "passed") if field in value}
                      for key, value in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
