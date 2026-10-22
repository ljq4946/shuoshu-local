from __future__ import annotations

import copy
import math
import os
import shutil
from pathlib import Path

from .quality import normalized_critical_terms
from .storage import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
INDEX_REVISION = "d9e41aac89fd00b3d71497fddb287b7f24613712"
DEFAULTS = {
    "tts_python": ".runtime/index-tts/.venv/Scripts/python.exe",
    "asr_python": ".runtime/asr/Scripts/python.exe",
    "index_repo": ".runtime/index-tts",
    "index_model": ".runtime/models/index-tts-2.5",
    "tts_adapter": "",
    "asr_model": ".runtime/models/sensevoice-small",
    "asr_vad_model": ".runtime/models/fsmn-vad",
    "ffmpeg": "",
    "device": "cuda:0",
    "bf16": True,
    "speed": 1.0,
    "inference_mode": "legacy",
    "inference_gap_ms": None,
    "max_chars": 180,
    "max_retries": 2,
    "stage_timeout_seconds": 7200,
    "gap_ms": 180,
    "scene_gap_ms": 800,
    "qa": {"max_cer": 0.08, "max_gap": 4, "min_chars_per_second": 0.5, "max_chars_per_second": 14.0,
           "critical_terms": []},
    "export": {"lufs": -19.0, "true_peak": -1.5, "lra": 11.0, "mp3_bitrate": "128k"},
}
BOOK_SETTINGS = ("bf16", "speed", "inference_mode", "inference_gap_ms", "max_chars", "max_retries", "stage_timeout_seconds",
                 "gap_ms", "scene_gap_ms", "qa", "export", "tts_adapter")


def load_config(path: Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None and not path.is_file():
        raise ValueError(f"指定的配置文件不存在: {path}")
    path = path or ROOT / "config.local.json"
    if path.exists():
        supplied = read_json(path)
        if not isinstance(supplied, dict):
            raise ValueError("配置必须是 JSON 对象。")
        unknown = set(supplied) - set(cfg)
        if unknown:
            raise ValueError(f"未知配置项: {', '.join(sorted(unknown))}")
        for key, value in supplied.items():
            if isinstance(cfg[key], dict):
                if not isinstance(value, dict) or set(value) - set(cfg[key]):
                    raise ValueError(f"无效配置: {key}")
                cfg[key].update(value)
            else:
                cfg[key] = value
    for key in ("tts_python", "asr_python", "index_repo", "index_model", "asr_model", "asr_vad_model"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    if not isinstance(cfg["tts_adapter"], str):
        raise ValueError("tts_adapter 必须为空字符串或本地适配器路径。")
    if cfg["tts_adapter"]:
        cfg["tts_adapter"] = str((ROOT / cfg["tts_adapter"]).resolve())
    if not cfg["ffmpeg"]:
        candidates = [ROOT / ".runtime/bin/ffmpeg.exe"] if (ROOT / ".runtime/bin/ffmpeg.exe").is_file() else []
        candidates += sorted((ROOT / ".tools/bootstrap/imageio_ffmpeg/binaries").glob("ffmpeg*.exe"))
        cfg["ffmpeg"] = str(candidates[0]) if candidates else (shutil.which("ffmpeg") or "")
    elif not Path(cfg["ffmpeg"]).is_absolute():
        cfg["ffmpeg"] = str((ROOT / cfg["ffmpeg"]).resolve())
    validate_config(cfg)
    cfg["inference_gap_ms"] = inference_gap(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    numeric = lambda value: type(value) in (int, float) and math.isfinite(value)
    if not isinstance(cfg["tts_adapter"], str):
        raise ValueError("tts_adapter 必须为空字符串或本地适配器路径。")
    if cfg["tts_adapter"] and not Path(cfg["tts_adapter"]).is_file():
        raise ValueError(f"音色适配器不存在: {cfg['tts_adapter']}")
    if not numeric(cfg["speed"]) or not 0.5 <= cfg["speed"] <= 2.0:
        raise ValueError("speed 必须在 0.5–2.0 之间（大于 1 表示放慢）。")
    if cfg.get("inference_mode", "legacy") not in {"legacy", "sentence"}:
        raise ValueError("inference_mode 必须是 legacy 或 sentence。")
    internal_gap = cfg.get("inference_gap_ms")
    if internal_gap is not None and (type(internal_gap) is not int or not 0 <= internal_gap <= 3000):
        raise ValueError("inference_gap_ms 必须为空或 0–3000 的整数。")
    if type(cfg["max_chars"]) is not int or not 20 <= cfg["max_chars"] <= 500:
        raise ValueError("max_chars 必须为 20–500 的整数。")
    if type(cfg["max_retries"]) is not int or cfg["max_retries"] != 2:
        raise ValueError("自动重试固定为 2 次；人工重做请使用 retry。")
    qa = cfg["qa"]
    if (not isinstance(qa.get("critical_terms", []), list)
            or any(not isinstance(term, str) or not term for term in qa.get("critical_terms", []))):
        raise ValueError("qa.critical_terms 必须是非空关键字组成的列表。")
    normalized_critical_terms(qa.get("critical_terms", []))
    if (not numeric(qa["max_cer"]) or not 0 <= qa["max_cer"] <= 1
            or type(qa["max_gap"]) is not int or qa["max_gap"] < 1
            or not numeric(qa["min_chars_per_second"]) or not numeric(qa["max_chars_per_second"])
            or not 0 < qa["min_chars_per_second"] <= qa["max_chars_per_second"]):
        raise ValueError("无效的校对阈值。")
    if (type(cfg["gap_ms"]) is not int or not 0 <= cfg["gap_ms"] <= 3000
            or type(cfg["scene_gap_ms"]) is not int or not 0 <= cfg["scene_gap_ms"] <= 10000
            or not numeric(cfg["stage_timeout_seconds"]) or cfg["stage_timeout_seconds"] <= 0):
        raise ValueError("无效的段间停顿或阶段超时。")


def inference_gap(cfg: dict) -> int:
    gap = cfg.get("inference_gap_ms")
    if gap is None:
        return cfg.get("gap_ms", DEFAULTS["gap_ms"]) if cfg.get("inference_mode", "legacy") == "sentence" else 200
    return gap


def book_inference_gap(project: dict) -> int:
    # Old sentence books used their saved gap; legacy inference used 200 ms.
    saved = {**project.get("book_config", {}), "inference_mode": project["profile"].get("inference_mode", "legacy")}
    return inference_gap(saved)


def book_settings(cfg: dict) -> dict:
    result = {key: copy.deepcopy(cfg[key]) for key in BOOK_SETTINGS}
    result["inference_gap_ms"] = inference_gap(cfg)
    return result


def load_book_config(root: Path, path: Path | None = None) -> dict:
    """Restore production settings; resolve runtime paths on this installation."""
    cfg = load_config(path)
    project = read_json(root / "project.json")
    saved = project.get("book_config", {})
    # Older books still have a frozen speed/precision profile.
    saved = {**{key: project["profile"][key] for key in ("speed", "bf16")}, **saved}
    for key in BOOK_SETTINGS:
        if key in saved:
            cfg[key] = copy.deepcopy(saved[key])
    cfg["inference_gap_ms"] = book_inference_gap(project)
    if path is not None:
        for key, value in read_json(path).items():
            if key in BOOK_SETTINGS:
                if key == "inference_gap_ms" and value is None:
                    continue  # Auto on an existing book means its frozen value.
                if isinstance(cfg[key], dict):
                    cfg[key].update(value)
                else:
                    cfg[key] = value
    validate_config(cfg)
    cfg["inference_gap_ms"] = inference_gap(cfg)
    return cfg


def offline_environment() -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "MODELSCOPE_OFFLINE": "1",
        "HF_HOME": str(ROOT / ".runtime/cache/huggingface"),
        "NLTK_DATA": str(ROOT / ".runtime/nltk_data"),
        "PYTHONPATH": str(ROOT),
    })
    return env


def generation_profile(cfg: dict) -> dict:
    # A saved project never silently changes voice engines/settings on resume.
    profile = {"engine": "IndexTTS-2.5", "revision": INDEX_REVISION,
               **{key: cfg[key] for key in ("bf16", "speed")}}
    if cfg.get("inference_mode", "legacy") != "legacy":
        profile["inference_mode"] = cfg["inference_mode"]
    if cfg.get("tts_adapter"):
        from .storage import file_hash
        profile["adapter_sha256"] = file_hash(Path(cfg["tts_adapter"]))
    return profile


def asset_signature(cfg: dict) -> str:
    records = []
    for key in ("index_model", "asr_model", "asr_vad_model"):
        root = Path(cfg[key])
        for path in sorted(root.rglob("*")) if root.is_dir() else []:
            if path.is_file() and path.name != "asset-manifest.json" and ".cache" not in path.parts and "__pycache__" not in path.parts:
                stat = path.stat()
                records.append((key, str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    source = Path(cfg["index_repo"]) / "indextts/infer_v2_5.py"
    if source.is_file():
        from .storage import file_hash
        records.append(("index_code", file_hash(source)))
    if cfg.get("tts_adapter"):
        from .storage import file_hash
        records.append(("tts_adapter", file_hash(Path(cfg["tts_adapter"]))))
    return digest(records)
