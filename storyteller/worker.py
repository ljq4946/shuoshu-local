"""One model per process. All heavy imports stay out of the controller."""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

from .config import inference_gap
from .runtime import PREFIX, block_network
from .storage import read_json
from .text import split_inference_text


def emit(event: dict):
    sys.__stdout__.write(PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    sys.__stdout__.flush()


def tts_model(cfg):
    sys.path.insert(0, cfg["index_repo"])
    os.chdir(cfg["index_repo"])
    import torch
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。检查 NVIDIA 驱动与 PyTorch CUDA 安装。")
    from .compat import prepare_wetext
    prepare_wetext()
    from indextts.infer_v2_5 import IndexTTS2
    class LocalIndexTTS(IndexTTS2):
        # Upstream splits at every period before normalizing the text. Keep its
        # low-VRAM path while protecting decimals, times and G2P annotations.
        split_text_by_punctuation = staticmethod(split_inference_text)

        def split_text_by_tokens(self, *args, **kwargs):
            parts = super().split_text_by_tokens(*args, **kwargs)
            if hasattr(self, "narration_chunks"):
                self.narration_chunks.extend(parts)
            return parts

    root = Path(cfg["index_model"])
    config = root / "config_v2_5.yaml"
    if not config.exists():
        config = root / "config.yaml"
    model = LocalIndexTTS(cfg_path=str(config), model_dir=str(root), device=cfg["device"],
                          use_bf16=cfg["bf16"], use_deepspeed=False, use_cuda_kernel=False,
                          use_accel=False, use_torch_compile=False, use_qwen_emo=False)
    if cfg.get("tts_adapter"):
        from .lora import merge_adapter
        from .storage import file_hash
        adapter = torch.load(cfg["tts_adapter"], map_location="cpu", weights_only=True)
        if adapter.get("base", {}).get("gpt_sha256") != file_hash(root / "gpt.pth"):
            raise ValueError("音色适配器与当前基础模型不匹配。")
        merge_adapter(model.gpt, adapter)
        print(f">> Loaded local voice adapter: {cfg['tts_adapter']}", file=sys.stderr)
    return model


def asr_model(cfg):
    from funasr import AutoModel
    from .compat import unicode_sentencepiece
    # Integrated, version-pinned model code; no remote-code downloads.
    with unicode_sentencepiece():
        return AutoModel(model=cfg["asr_model"], vad_model=cfg["asr_vad_model"],
                         vad_kwargs={"max_single_segment_time": 30000}, device=cfg["device"],
                         trust_remote_code=False, disable_update=True, disable_pbar=True,
                         ncpu=2)


def synthesize(model, cfg, reference, text, output):
    model.narration_chunks = []
    options = dict(spk_audio_prompt=str(reference), text=text, lang="ZH", output_path=str(output),
                   duration_factor=cfg["speed"], use_random=False, use_emo_text=False,
                   max_text_tokens_per_segment=120, verbose=False,
                   interval_silence=inference_gap(cfg))
    if cfg.get("inference_mode", "legacy") == "sentence":
        # Preserve sentence context instead of the upstream 40-character pass.
        # Normalized tokenizer splitting remains enabled as a memory safeguard.
        for _ in model.infer_generator(**options):
            pass
    else:
        model.infer(**options)
    return model.narration_chunks


def release_sentence_cuda_cache(cfg):
    if cfg.get("inference_mode", "legacy") != "sentence" or not cfg["device"].startswith("cuda"):
        return None
    import gc
    import torch
    before = {"allocated_mib": torch.cuda.memory_allocated() / 2**20,
              "reserved_mib": torch.cuda.memory_reserved() / 2**20}
    gc.collect()
    # Release unused allocator blocks between controlled long-text jobs. On an
    # 8 GiB Windows GPU, retained high-water blocks can trigger WDDM paging and
    # make following jobs dramatically slower despite high GPU utilization.
    torch.cuda.empty_cache()
    after = {"allocated_mib": torch.cuda.memory_allocated() / 2**20,
             "reserved_mib": torch.cuda.memory_reserved() / 2**20}
    return {"before": before, "after": after}


def main(request: Path) -> int:
    spec = read_json(request)
    root, cfg, stage = Path(spec["root"]), spec["config"], spec["stage"]
    block_network()
    with contextlib.redirect_stdout(sys.stderr):
        model = tts_model(cfg) if stage == "synth" else asr_model(cfg)
        emit({"event": "ready", "stage": stage})
        for job in spec["jobs"]:
            emit({"event": "started", "id": job["id"]})
            started = time.perf_counter()
            try:
                if stage == "synth":
                    import numpy as np
                    import soundfile as sf
                    import torch
                    random.seed(job["seed"])
                    np.random.seed(job["seed"] % (2**32))
                    torch.manual_seed(job["seed"])
                    torch.cuda.manual_seed_all(job["seed"])
                    output = Path(job["output"])
                    output.parent.mkdir(exist_ok=True)
                    temp = output.with_suffix(".partial.wav")
                    inference_texts = synthesize(model, cfg, root / "input/reference.wav", job["text"], temp)
                    data, rate = sf.read(temp, dtype="float32", always_2d=True)
                    if not len(data) or not np.isfinite(data).all():
                        raise RuntimeError("模型返回了空音频或非法采样值。")
                    sf.write(str(temp), data.mean(axis=1), rate, subtype="PCM_16")
                    os.replace(temp, output)
                    cuda_cache = release_sentence_cuda_cache(cfg)
                    emit({"event": "done", "id": job["id"], "seconds": time.perf_counter() - started,
                          "inference_mode": cfg.get("inference_mode", "legacy"), "inference_texts": inference_texts,
                          "cuda_cache": cuda_cache})
                else:
                    result = model.generate(input=job["audio"], cache={}, language="zh", use_itn=True,
                                            batch_size_s=30, merge_vad=False)
                    # Keep the raw transcription for audit; strip model tags, not words.
                    import re
                    transcript = "".join(item.get("text", "") for item in result)
                    transcript = re.sub(r"<\|[^>]*\|>", "", transcript)
                    emit({"event": "done", "id": job["id"], "recognized": transcript,
                          "seconds": time.perf_counter() - started})
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit({"event": "error", "id": job["id"], "error": f"{type(exc).__name__}: {exc}",
                      "seconds": time.perf_counter() - started})
                # Recovering from CUDA OOM within the same model can poison later jobs.
                if "out of memory" in str(exc).lower():
                    return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(Path(sys.argv[1])))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
