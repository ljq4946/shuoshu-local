from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .config import ROOT, offline_environment
from .storage import write_json

PREFIX = "@@SHUOSHU@@"


def prerequisites(cfg: dict) -> list[str]:
    missing = []
    for key in ("tts_python", "asr_python", "ffmpeg"):
        if not cfg[key] or not Path(cfg[key]).is_file():
            missing.append(f"{key}: {cfg[key] or '未配置'}")
    if not (Path(cfg["index_repo"]) / "indextts/infer_v2_5.py").is_file():
        missing.append("IndexTTS 2.5 源码未安装")
    for key in ("index_model", "asr_model", "asr_vad_model"):
        manifest = Path(cfg[key]) / "asset-manifest.json"
        if not manifest.is_file():
            missing.append(f"{key}: 尚未准备完整模型（缺少 asset-manifest.json）")
        else:
            from .storage import read_json
            try:
                for item in read_json(manifest)["files"]:
                    from .storage import contained
                    path = contained(Path(cfg[key]), item["path"])
                    if not path.is_file() or path.stat().st_size != item["bytes"]:
                        missing.append(f"模型文件缺失或大小改变: {path}")
                if not read_json(manifest)["files"]:
                    missing.append(f"模型清单为空: {manifest}")
            except (ValueError, KeyError, OSError) as exc:
                missing.append(f"模型清单无效: {manifest}: {exc}")
    required = {
        "index_model": ["config.yaml", "gpt.pth", "codec.pth", "s2mel.pth", "wav2vec2bert_stats.pt",
                        "feat1.pt", "feat2.pt", "multilingual_zh_ja_yue_char_del.tiktoken",
                        "hf_cache/campplus_cn_common.bin", "hf_cache/w2v-bert-2.0/config.json",
                        "hf_cache/w2v-bert-2.0/model.safetensors", "hf_cache/bigvgan/config.json",
                        "hf_cache/bigvgan/bigvgan_generator.pt"],
        "asr_model": ["config.yaml", "model.pt"], "asr_vad_model": ["config.yaml", "model.pt"],
    }
    for key, names in required.items():
        for name in names:
            if not (Path(cfg[key]) / name).is_file():
                missing.append(f"缺少必要模型文件: {key}/{name}")
    return missing


class StageRunner:
    def __init__(self, cfg: dict, cancel_event=None):
        self.cfg = cfg
        self.cancel_event = cancel_event

    def check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise KeyboardInterrupt("已暂停制作，进度已保存。")

    def events(self, stage: str, root: Path, jobs: list[dict]):
        self.check_cancelled()
        token = uuid.uuid4().hex
        jobs_dir = root / ".jobs"
        jobs_dir.mkdir(exist_ok=True)
        request = jobs_dir / f"{stage}-{token}.json"
        log = root / "logs" / f"{stage}-{token}.log"
        log.parent.mkdir(exist_ok=True)
        write_json(request, {"stage": stage, "root": str(root), "config": self.cfg, "jobs": jobs})
        python = self.cfg["tts_python" if stage == "synth" else "asr_python"]
        command = [python, "-u", "-m", "storyteller.worker", str(request)]
        env = offline_environment()
        if self.cfg["ffmpeg"]:
            env["PATH"] = str(Path(self.cfg["ffmpeg"]).parent) + os.pathsep + env.get("PATH", "")
        process = None
        timed_out = threading.Event()
        with log.open("w", encoding="utf-8") as errors:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=errors,
                                           text=True, encoding="utf-8", errors="replace",
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                stopped = threading.Event()
                activity = [time.monotonic()]
                limit = self.cfg["stage_timeout_seconds"]
                def watchdog():
                    while not stopped.wait(min(1.0, limit / 2)):
                        if self.cancel_event is not None and self.cancel_event.is_set():
                            if process.poll() is None:
                                process.terminate()
                            return
                        if time.monotonic() - activity[0] > limit:
                            timed_out.set()
                            if process.poll() is None:
                                process.kill()
                            return
                watcher = threading.Thread(target=watchdog, daemon=True)
                watcher.start()
                try:
                    for line in process.stdout:
                        if line.startswith(PREFIX):
                            activity[0] = time.monotonic()
                            yield json.loads(line[len(PREFIX):])
                        else:
                            errors.write(line)
                    code = process.wait()
                    self.check_cancelled()
                    if code:
                        reason = "阶段长时间无进度，已超时" if timed_out.is_set() else f"进程退出码 {code}"
                        raise RuntimeError(f"{stage} {reason}，已保存进度。日志：{log}")
                finally:
                    stopped.set()
                    watcher.join(timeout=2)
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    if process.stdout:
                        process.stdout.close()


def block_network() -> None:
    """Inference workers must never fetch missing assets or call a hosted API."""
    import socket
    def denied(*args, **kwargs):
        raise OSError("离线模式禁止网络访问；请先运行 setup.ps1 -Models 准备全部依赖。")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    socket.getaddrinfo = denied
