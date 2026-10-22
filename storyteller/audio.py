from __future__ import annotations

import array
import json
import math
import os
import re
import subprocess
import sys
import wave
from pathlib import Path


def run_ffmpeg(executable: str, args: list[str]) -> str:
    if not executable or not Path(executable).is_file():
        raise RuntimeError("未找到 FFmpeg，请先运行 setup.ps1 或配置 ffmpeg 路径。")
    result = subprocess.run([executable, "-hide_banner", "-nostdin", "-y", *args],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError("FFmpeg 失败：" + result.stderr[-3000:])
    return result.stderr


def wav_info(path: Path) -> dict:
    squares = count = peaks = 0
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise ValueError(f"音频必须是 16 位单声道 WAV: {path.name}")
        rate, frames = stream.getframerate(), stream.getnframes()
        while raw := stream.readframes(65536):
            samples = array.array("h", raw)
            if sys.byteorder != "little":
                samples.byteswap()
            squares += sum(sample * sample for sample in samples)
            peaks += sum(abs(sample) >= 32760 for sample in samples)
            count += len(samples)
    if count != frames:
        raise ValueError(f"WAV 文件不完整: {path.name}")
    rms = math.sqrt(squares / max(1, count)) / 32768
    return {"duration": frames / rate, "sample_rate": rate, "frames": frames,
            "rms_db": max(-120.0, 20 * math.log10(max(1e-6, rms))),
            "clipped_fraction": peaks / max(1, count)}


def prepare_reference(source: Path, destination: Path, ffmpeg: str, start: float = 0) -> dict:
    if start < 0:
        raise ValueError("参考音频起点不能为负数。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(ffmpeg, ["-ss", str(start), "-i", str(source), "-t", "15", "-vn", "-ac", "1",
                        "-ar", "24000", "-c:a", "pcm_s16le", str(destination)])
    info = wav_info(destination)
    if info["duration"] < 3:
        raise ValueError("参考录音不足 3 秒；建议提供 10–15 秒清晰单人录音。")
    if info["rms_db"] < -50:
        raise ValueError("参考录音近乎静音，请更换录音。")
    return {**info, "start_seconds": start, "max_seconds": 15}


def concatenate(paths: list[Path | int], output: Path, gap_ms: int) -> int:
    """Integer entries are explicit scene pauses, replacing the ordinary gap."""
    voices = [path for path in paths if isinstance(path, Path)]
    if not voices:
        raise ValueError("没有可合并音频。")
    with wave.open(str(voices[0]), "rb") as first:
        rate = first.getframerate()
    with wave.open(str(output), "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        for index, path in enumerate(paths):
            if isinstance(path, int):
                if not 0 <= path <= 10000:
                    raise ValueError("场景停顿必须在 0–10000 毫秒之间。")
                target.writeframesraw(b"\0\0" * int(rate * path / 1000))
                continue
            with wave.open(str(path), "rb") as source:
                if source.getnchannels() != 1 or source.getsampwidth() != 2:
                    raise ValueError("合并音频必须全部为 PCM16 单声道。")
                if source.getframerate() != rate:
                    raise ValueError("各片段采样率不一致。")
                if index and isinstance(paths[index-1], Path):
                    target.writeframesraw(b"\0\0" * int(rate * gap_ms / 1000))
                while block := source.readframes(65536):
                    target.writeframesraw(block)
    return rate


def export_chapter(paths: list[Path | int], directory: Path, name: str, title: str, cfg: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / f"{name}.concat.tmp.wav"
    master_tmp, mp3_tmp = directory / f"{name}.tmp.wav", directory / f"{name}.tmp.mp3"
    master, mp3 = directory / f"{name}.wav", directory / f"{name}.mp3"
    try:
        rate = concatenate(paths, raw, cfg["gap_ms"])
        settings = cfg["export"]
        loudnorm = f"loudnorm=I={settings['lufs']}:TP={settings['true_peak']}:LRA={settings['lra']}"
        log = run_ffmpeg(cfg["ffmpeg"], ["-i", str(raw), "-af", loudnorm + ":print_format=json", "-f", "null", "-"])
        matches = re.findall(r'\{\s*"input_i"[\s\S]*?\}', log)
        if not matches:
            raise RuntimeError("未取得响度分析结果。")
        measured = json.loads(matches[-1])
        if not all(math.isfinite(float(measured[key])) for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")):
            raise RuntimeError("章节响度不可计算，可能没有有效语音。")
        second = (loudnorm + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
                  f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
                  f":offset={measured['target_offset']}:linear=true:print_format=json")
        run_ffmpeg(cfg["ffmpeg"], ["-i", str(raw), "-af", second, "-ar", str(rate), "-ac", "1",
                                  "-c:a", "pcm_s16le", str(master_tmp)])
        run_ffmpeg(cfg["ffmpeg"], ["-i", str(master_tmp), "-c:a", "libmp3lame", "-b:a", settings["mp3_bitrate"],
                                  "-metadata", f"title={title}", str(mp3_tmp)])
        os.replace(master_tmp, master)
        os.replace(mp3_tmp, mp3)
        return {"wav": master.name, "mp3": mp3.name, "input_lufs": float(measured["input_i"]),
                "target_lufs": settings["lufs"], "duration": wav_info(master)["duration"]}
    finally:
        for temp in (raw, master_tmp, mp3_tmp):
            temp.unlink(missing_ok=True)
