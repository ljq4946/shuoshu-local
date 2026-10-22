"""Local IndexTTS 2.5 voice-adapter preparation, training and evaluation.

Run with the existing IndexTTS virtualenv. Heavy imports stay inside commands.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

from .config import INDEX_REVISION, ROOT, load_config, offline_environment
from .training_data import dump_json, read_jsonl, prepare_dataset, sha256, verify_sources
from .storage import FileLock


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def setup_runtime():
    cfg = load_config()
    os.environ.update(offline_environment())
    sys.path.insert(0, cfg["index_repo"])
    from .runtime import block_network
    block_network()
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local training profile")
    return cfg


def provenance(cfg):
    root = Path(cfg["index_model"])
    manifest = json.loads((root / "asset-manifest.json").read_text(encoding="utf-8"))
    for name in ("gpt.pth", "codec.pth", "config.yaml", "wav2vec2bert_stats.pt", "multilingual_zh_ja_yue_char_del.tiktoken"):
        expected = next(entry["sha256"] for entry in manifest["files"] if entry["path"] == name)
        if sha256(root / name) != expected:
            raise ValueError(f"Base model integrity check failed: {name}")
    return {"index_revision": INDEX_REVISION, "asset_manifest_sha256": sha256(root / "asset-manifest.json"),
            "gpt_sha256": sha256(root / "gpt.pth"), "base_revision": manifest["revisions"]["IndexTeam/IndexTTS-2.5"]}


def load_gpt(cfg, device="cpu", dtype=None):
    import torch
    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice
    model_root = Path(cfg["index_model"])
    spec = OmegaConf.load(model_root / "config.yaml")
    model = UnifiedVoice(**spec.gpt, spk_cond_mode="campplus", use_accel=False)
    state = torch.load(model_root / "gpt.pth", map_location="cpu", weights_only=True, mmap=True)
    state = state.get("model", state)
    result = model.load_state_dict(state, strict=False)
    # Official inference also ignores unused training keys; missing weights are fatal.
    if result.missing_keys:
        raise ValueError(f"Missing base GPT weights: {result.missing_keys}")
    if result.unexpected_keys:
        print("Unused base checkpoint keys:", result.unexpected_keys[:8], flush=True)
    del state
    gc.collect()
    model.requires_grad_(False).eval()
    return model.to(device=device, dtype=dtype), spec


def dataset_report(dataset):
    report = json.loads((dataset / "dataset_report.json").read_text(encoding="utf-8"))
    for split, expected in report["manifest_sha256"].items():
        if sha256(dataset / f"{split}.jsonl") != expected:
            raise ValueError(f"Prepared manifest changed: {split}")
    return report


def atomic_torch_save(value, destination):
    import torch
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(20):
        try:
            temporary.replace(destination)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.05)


def extract(dataset: Path, limit: int | None = None):
    cfg = setup_runtime()
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from omegaconf import OmegaConf
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
    from indextts.codec.models import EnhancedCodec
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from indextts.utils.front import TextNormalizer
    from indextts.utils.ja_g2p import JapaneseG2PProcessor
    from indextts.utils.nemo_tn import normalize_text
    from indextts.utils.tokenizer import get_tokenizer, lang_to_token
    report = dataset_report(dataset)
    source_check = verify_sources(dataset)
    base = provenance(cfg)
    fingerprint = {"version": 1, "source_fingerprint": report["source_fingerprint"], "base": base,
                   "text_frontend": "upstream-char-map+ja-TN+fugashi-g2p0+ja-prefix", "reference_conditioning": "frozen target-utterance speaker/emotion"}
    feature_dir = dataset / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    meta_file = feature_dir / "metadata.json"
    if meta_file.exists() and json.loads(meta_file.read_text(encoding="utf-8")) != fingerprint:
        raise ValueError("Existing feature cache belongs to different source data or model")
    dump_json(meta_file, fingerprint)
    index_file = feature_dir / "index.json"
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else {}
    rows = [r for split in ("train", "validation", "test") for r in read_jsonl(dataset / f"{split}.jsonl")]
    if limit:
        rows = rows[:limit]
    pending = []
    for row in rows:
        path = feature_dir / (row["id"] + ".pt")
        record = index.get(row["id"], {})
        if not path.is_file() or sha256(path) != record.get("sha256"):
            pending.append(row)
    status_file = dataset / "extraction_status.json"
    start = time.perf_counter()
    dump_json(status_file, {"status": "loading_models", "total": len(rows), "cached": len(rows) - len(pending), "updated_at": timestamp()})
    if not pending:
        dump_json(status_file, {"status": "complete", "total": len(rows), "cached": len(rows), "updated_at": timestamp()})
        return
    model_root = Path(cfg["index_model"])
    spec = OmegaConf.load(model_root / "config.yaml")
    w2v_root = model_root / "hf_cache/w2v-bert-2.0"
    frontend = SeamlessM4TFeatureExtractor.from_pretrained(str(w2v_root), local_files_only=True)
    semantic_model = Wav2Vec2BertModel.from_pretrained(str(w2v_root), local_files_only=True).eval().to("cuda")
    codec = EnhancedCodec(**spec.semantic_codec, cfg=spec.semantic_codec)
    codec.load_checkpoint(str(model_root / "codec.pth"))
    codec.requires_grad_(False).eval().to("cuda")
    stats = torch.load(model_root / "wav2vec2bert_stats.pt", map_location="cpu", weights_only=True)
    mean, std = stats["mean"].to("cuda"), stats["var"].sqrt().to("cuda")
    camp = CAMPPlus(feat_dim=80, embedding_size=192)
    camp.load_state_dict(torch.load(model_root / "hf_cache/campplus_cn_common.bin", map_location="cpu", weights_only=True))
    camp.requires_grad_(False).eval().to("cuda")
    gpt, _ = load_gpt(cfg)
    for name in ("spk_emb_proj", "emo_conditioning_encoder", "emo_perceiver_encoder", "emovec_layer", "emo_layer"):
        getattr(gpt, name).to("cuda")
    tokenizer = get_tokenizer(multilingual=True, model_dir=str(model_root))
    normalizer, ja = TextNormalizer(), JapaneseG2PProcessor(g2p_ratio=0)
    for number, row in enumerate(pending, 1):
        with torch.inference_mode():
            if sha256(Path(row["audio"])) != row["audio_sha256"]:
                raise ValueError(f"Source changed: {row['id']}")
            audio, rate = sf.read(row["audio"], dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
            audio = torchaudio.functional.resample(audio, rate, 16000)
            inputs = frontend(audio.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
            output = semantic_model(input_features=inputs["input_features"].to("cuda"),
                                    attention_mask=inputs["attention_mask"].to("cuda"), output_hidden_states=True)
            semantic = (output.hidden_states[17] - mean) / std
            codes, _ = codec.quantize(semantic)
            codes = codes.reshape(-1).long()
            fbank = torchaudio.compliance.kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
            fbank -= fbank.mean(dim=0, keepdim=True)
            style = camp(fbank.unsqueeze(0).to("cuda"))
            emotion = gpt.get_emovec(semantic, torch.tensor([semantic.shape[1]], device="cuda"))
            projected = gpt.spk_emb_proj(style).unsqueeze(1) + emotion.unsqueeze(1)
            conditioning = torch.cat((projected, torch.zeros(1, 2, projected.shape[-1], device="cuda")), dim=1)
            text = normalizer.clean_pattern.sub(lambda match: normalizer.char_rep_map[match.group()], row["text"])
            processed = ja.process_ja_text(normalize_text(text.lower(), "ja"))
            token_ids = tokenizer.encode("<|ja|> " + processed, allowed_special="all")
            if not token_ids or len(token_ids) > 580 or not codes.numel() or codes.numel() > 1800:
                raise ValueError(f"Invalid token lengths: {row['id']}")
            if codes.min() < 0 or codes.max() >= 8192:
                raise ValueError(f"Invalid codec token: {row['id']}")
            feature = {"id": row["id"], "audio_sha256": row["audio_sha256"], "text": row["text"], "processed_text": processed,
                       "text_tokens": torch.tensor(token_ids, dtype=torch.long), "codes": codes.cpu(),
                       "conditioning": conditioning.cpu().float(), "language_id": lang_to_token("JA")}
            atomic_torch_save(feature, feature_dir / (row["id"] + ".pt"))
            index[row["id"]] = {"sha256": sha256(feature_dir / (row["id"] + ".pt")), "text_tokens": len(token_ids), "code_tokens": int(codes.numel())}
            del output, semantic, codes, conditioning
        if number % 20 == 0 or number == len(pending):
            dump_json(index_file, index)
            elapsed = time.perf_counter() - start
            status = {"status": "complete" if number == len(pending) else "extracting", "total": len(rows),
                      "done": len(rows) - len(pending) + number, "elapsed_seconds": elapsed, "updated_at": timestamp(),
                      "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20, "source_check": source_check}
            dump_json(status_file, status)
            print(json.dumps(status), flush=True)


def load_feature(dataset, item_id, device="cpu"):
    import torch
    return torch.load(dataset / "features" / (item_id + ".pt"), map_location=device, weights_only=True)


def semantic_loss(model, feature):
    """Teacher-forced code prediction, with exactly the inference prefix layout."""
    import torch
    from torch.nn import functional as F
    device = next(model.parameters()).device
    text = feature["text_tokens"].to(device).reshape(1, -1)
    codes = feature["codes"].to(device).reshape(1, -1)
    cond = feature["conditioning"].to(device=device, dtype=model.text_embedding.weight.dtype)
    langs = torch.tensor([feature["language_id"]], device=device)
    _, prefix, _ = model.prepare_gpt_inputs(cond, text, langs)
    mel_inputs = F.pad(codes, (1, 0), value=model.start_mel_token)
    mel_embeddings = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)
    embeddings = torch.cat((prefix, mel_embeddings), dim=1)
    output = model.gpt(inputs_embeds=embeddings, use_cache=False, return_dict=True)
    hidden = output.last_hidden_state[:, -mel_inputs.shape[1]:]
    logits = model.mel_head(model.final_norm(hidden))
    targets = F.pad(codes, (0, 1), value=model.stop_mel_token)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss, targets.numel()


def evaluate_loss(model, features):
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for feature in features:
            loss, tokens = semantic_loss(model, feature)
            total += float(loss) * tokens
            count += tokens
    return {"loss": total / count, "tokens": count, "samples": len(features)}


def train(dataset: Path, run: Path, epochs=3, rank=16, learning_rate=1e-4, accumulation=8, resume=False, max_updates=None):
    cfg = setup_runtime()
    import torch
    from .lora import adapter_state, inject_lora, load_adapter_state
    torch.manual_seed(20261022)
    random.seed(20261022)
    report = dataset_report(dataset)
    base = provenance(cfg)
    sources = verify_sources(dataset)
    metadata = json.loads((dataset / "features/metadata.json").read_text(encoding="utf-8"))
    if metadata["base"] != base or metadata["source_fingerprint"] != report["source_fingerprint"]:
        raise ValueError("Feature cache provenance mismatch")
    feature_index = json.loads((dataset / "features/index.json").read_text(encoding="utf-8"))
    splits = {split: read_jsonl(dataset / f"{split}.jsonl") for split in ("train", "validation", "test")}
    for rows in splits.values():
        for row in rows:
            path = dataset / "features" / (row["id"] + ".pt")
            if row["id"] not in feature_index or sha256(path) != feature_index[row["id"]]["sha256"]:
                raise ValueError(f"Missing or corrupt feature cache: {row['id']}")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "last.pt").exists() and not resume:
        raise ValueError("Run already exists. Use --resume or a new run name")
    lora_config = {"rank": rank, "alpha": float(rank * 2), "dropout": 0.05}
    config = {"epochs": epochs, "learning_rate": learning_rate, "accumulation": accumulation, "seed": 20261022,
              "precision": "bfloat16", "base": base, "dataset_fingerprint": report["source_fingerprint"],
              "lora_config": lora_config, "source_check": sources}
    old_config_path = run / "config.json"
    if resume and old_config_path.exists() and json.loads(old_config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume configuration differs from the saved run")
    dump_json(old_config_path, config)
    model, _ = load_gpt(cfg, device="cuda", dtype=torch.bfloat16)
    names = inject_lora(model, **lora_config)
    model.gpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    validation = [load_feature(dataset, r["id"]) for r in splits["validation"]]
    test = [load_feature(dataset, r["id"]) for r in splits["test"]]
    updates_per_epoch = math.ceil(len(splits["train"]) / accumulation)
    total_updates = epochs * updates_per_epoch
    best_loss, step, epoch_start, batch_start = float("inf"), 0, 0, 0
    baseline, baseline_test = None, None
    if resume:
        saved = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        load_adapter_state(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        best_loss, step = saved["best_validation_loss"], saved["step"]
        epoch_start, batch_start = saved["next_epoch"], saved["next_batch"]
        baseline, baseline_test = saved["baseline_validation"], saved["baseline_test"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device=0)
    if baseline is None:
        print("Evaluating untouched base on validation and test...", flush=True)
        baseline = evaluate_loss(model, validation)
        baseline_test = evaluate_loss(model, test)
        best_loss = baseline["loss"]
        dump_json(run / "baseline.json", {"validation": baseline, "test": baseline_test})
    trainable = sum(p.numel() for p in parameters)
    total = sum(p.numel() for p in model.parameters())
    started = time.perf_counter()
    status_path = run / "status.json"
    metrics_path = run / "metrics.jsonl"
    next_epoch, next_batch = epoch_start, batch_start

    def save_checkpoint(path, include_optimizer=False):
        checkpoint = {"format": "shuoshu-index25-lora-v1", "adapter": adapter_state(model), "lora_config": lora_config,
                      "base": base, "dataset_fingerprint": report["source_fingerprint"], "step": step,
                      "best_validation_loss": best_loss, "baseline_validation": baseline, "baseline_test": baseline_test,
                      "next_epoch": next_epoch, "next_batch": next_batch, "created_at": timestamp()}
        if include_optimizer:
            checkpoint.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        atomic_torch_save(checkpoint, path)

    # A valid baseline adapter exists even if every trained candidate regresses.
    if not (run / "best.pt").exists():
        save_checkpoint(run / "best.pt")
    try:
        for epoch in range(epoch_start, epochs):
            rows = list(splits["train"])
            random.Random(20261022 + epoch).shuffle(rows)
            groups = [rows[i:i + accumulation] for i in range(0, len(rows), accumulation)]
            for batch_index in range(batch_start if epoch == epoch_start else 0, len(groups)):
                model.eval()
                model.gpt.train()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                group = groups[batch_index]
                for row in group:
                    feature = load_feature(dataset, row["id"])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = semantic_loss(model, feature)
                        if not torch.isfinite(loss):
                            raise ValueError(f"Nonfinite loss: {row['id']}")
                    (loss / len(group)).backward()
                    losses.append(float(loss.detach()))
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                if not math.isfinite(grad_norm):
                    raise ValueError("Nonfinite adapter gradients")
                step += 1
                warmup = min(1.0, step / max(10, total_updates * 0.05))
                progress = step / total_updates
                rate = learning_rate * warmup * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * progress)))
                for group_config in optimizer.param_groups:
                    group_config["lr"] = rate
                optimizer.step()
                next_epoch, next_batch = epoch, batch_index + 1
                if next_batch == len(groups):
                    next_epoch, next_batch = epoch + 1, 0
                metric = {"epoch": epoch + 1, "step": step, "train_loss": sum(losses) / len(losses),
                          "learning_rate": rate, "grad_norm": grad_norm, "elapsed_seconds": time.perf_counter() - started}
                if step % 10 == 0:
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metric) + "\n")
                    status = {"status": "training", **metric, "total_updates": total_updates, "best_validation_loss": best_loss,
                              "trainable_parameters": trainable, "total_parameters": total, "projections": len(names),
                              "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20, "updated_at": timestamp()}
                    dump_json(status_path, status)
                    print(json.dumps(status), flush=True)
                if step % 50 == 0:
                    save_checkpoint(run / "last.pt", True)
                if max_updates and step >= max_updates:
                    save_checkpoint(run / "last.pt", True)
                    dump_json(status_path, {"status": "stopped_at_requested_update", "step": step, "updated_at": timestamp()})
                    return
            scores = evaluate_loss(model, validation)
            if scores["loss"] < best_loss:
                best_loss = scores["loss"]
                save_checkpoint(run / "best.pt")
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "validation", "epoch": epoch + 1, "step": step, **scores}) + "\n")
            save_checkpoint(run / "last.pt", True)
            print("Validation", epoch + 1, scores, flush=True)
        best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        load_adapter_state(model, best["adapter"])
        heldout = evaluate_loss(model, test)
        final_validation = evaluate_loss(model, validation)
        verification = verify_sources(dataset)
        summary = {"status": "complete", "step": step, "epochs": epochs, "best_step": best["step"],
                   "best_checkpoint": str((run / "best.pt").resolve()), "best_checkpoint_sha256": sha256(run / "best.pt"),
                   "baseline_validation": baseline, "best_validation": final_validation, "baseline_test": baseline_test,
                   "best_test": heldout, "validation_relative_reduction": 1 - final_validation["loss"] / baseline["loss"],
                   "test_relative_reduction": 1 - heldout["loss"] / baseline_test["loss"],
                   "trainable_parameters": trainable, "total_parameters": total, "base_weights_frozen": True,
                   "source_check": verification, "elapsed_seconds": time.perf_counter() - started,
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
                   "human_listening": "pending", "chinese_improvement": "not_established_by_japanese_code_loss",
                   "production_default_changed": False, "updated_at": timestamp()}
        dump_json(run / "summary.json", summary)
        dump_json(status_path, summary)
        print(json.dumps(summary), flush=True)
    except BaseException:
        # Only completed optimizer updates are checkpointed. A partial accumulation
        # is discarded, so resume does not skip samples or apply half an update.
        save_checkpoint(run / "last.pt", True)
        dump_json(status_path, {"status": "interrupted_or_failed", "step": step, "next_epoch": next_epoch,
                               "next_batch": next_batch, "error": traceback.format_exc(), "updated_at": timestamp()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "extract", "train", "verify"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "training/qiong1_20261022")
    parser.add_argument("--audio", type=Path, default=ROOT / "01_语音")
    parser.add_argument("--labels", type=Path, default=ROOT / "02_标注/qiong1.list")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if args.command == "prepare":
        print(json.dumps(prepare_dataset(args.audio, args.labels, dataset), ensure_ascii=False, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_sources(dataset)))
    elif args.command == "extract":
        with FileLock(ROOT / ".runtime/gpu.lock"):
            extract(dataset, args.limit)
    else:
        with FileLock(ROOT / ".runtime/gpu.lock"):
            train(dataset, (args.run or dataset / "runs/lora_rank16").resolve(), args.epochs, args.rank,
                  args.learning_rate, args.accumulation, args.resume, args.max_updates)


if __name__ == "__main__":
    main()
