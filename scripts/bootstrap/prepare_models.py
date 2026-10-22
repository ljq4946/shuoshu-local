"""The only network-enabled model preparation entrypoint. Safe to resume."""
from __future__ import annotations

import os
import sys
import fnmatch
from urllib.parse import quote
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["HF_HOME"] = str(ROOT / ".runtime/cache/huggingface")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from storyteller.storage import FileLock, file_hash, now, read_json, write_json
from ranged_download import download as ranged_download


def main():
    from huggingface_hub import HfApi, snapshot_download
    revisions_path = ROOT / ".runtime/model-revisions.json"
    revisions = read_json(revisions_path) if revisions_path.exists() else {}
    api = HfApi()

    def download(repo, dest, allow=None, ignore=None):
        with FileLock(ROOT / '.runtime/model-revisions.lock'):
            revisions.update(read_json(revisions_path) if revisions_path.exists() else {})
            if repo not in revisions:
                revisions[repo] = api.model_info(repo).sha
                write_json(revisions_path, revisions)
        print(f"Preparing {repo} at {revisions[repo]}", flush=True)
        info = api.model_info(repo, revision=revisions[repo], files_metadata=True)
        mirrors = {'facebook/w2v-bert-2.0': 'AI-ModelScope/w2v-bert-2.0',
                   'nvidia/bigvgan_v2_22khz_80band_256x': 'nv-community/bigvgan_v2_22khz_80band_256x',
                   'FunAudioLLM/SenseVoiceSmall': 'iic/SenseVoiceSmall',
                   'funasr/fsmn-vad': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                   'funasr/campplus': 'iic/speech_campplus_sv_zh-cn_16k-common'}
        for item in info.siblings:
            name = item.rfilename
            if (allow and not any(fnmatch.fnmatch(name, pattern) for pattern in allow)) or (ignore and any(fnmatch.fnmatch(name, pattern) for pattern in ignore)):
                continue
            if item.lfs and item.size >= 64 * 1024 * 1024:
                local = Path(dest) / name
                model_id = mirrors.get(repo, repo)
                mirror = f'https://modelscope.cn/api/v1/models/{model_id}/repo?Revision=master&FilePath={quote(name, safe="")}'
                canonical = f'https://huggingface.co/{repo}/resolve/{revisions[repo]}/{quote(name, safe="/")}'
                try:
                    ranged_download(mirror, local, item.lfs.sha256, item.size)
                except Exception as exc:
                    print(f'ModelScope unavailable or different artifact ({type(exc).__name__}); using pinned Hugging Face revision.', flush=True)
                    ranged_download(canonical, local, item.lfs.sha256, item.size)
        snapshot_download(repo, revision=revisions[repo], local_dir=dest, allow_patterns=allow,
                          ignore_patterns=ignore, max_workers=4)

    models = ROOT / ".runtime/models"
    index = models / "index-tts-2.5"
    download("IndexTeam/IndexTTS-2.5", index, ignore=["qwen*/*", "*.md", ".gitattributes"])
    download("facebook/w2v-bert-2.0", index / "hf_cache/w2v-bert-2.0",
             allow=["config.json", "preprocessor_config.json", "model.safetensors", "LICENSE*"])
    download("funasr/campplus", index / "hf_cache", allow=["campplus_cn_common.bin"])
    download("nvidia/bigvgan_v2_22khz_80band_256x", index / "hf_cache/bigvgan",
             allow=["config.json", "bigvgan_generator.pt", "LICENSE*"])
    download("FunAudioLLM/SenseVoiceSmall", models / "sensevoice-small",
             ignore=["example/*", "*.onnx", "*.md", "*.py", ".gitattributes"])
    download("funasr/fsmn-vad", models / "fsmn-vad",
             ignore=["example/*", "*.onnx", "*.md", "*.py", ".gitattributes"])
    for path in (index, models / "sensevoice-small", models / "fsmn-vad"):
        print(f"Writing SHA-256 manifest: {path.name}", flush=True)
        files = []
        for item in sorted(path.rglob("*")):
            if item.is_file() and ".cache" not in item.parts and item.name != "asset-manifest.json":
                files.append({"path": item.relative_to(path).as_posix(), "bytes": item.stat().st_size,
                              "sha256": file_hash(item)})
        write_json(path / "asset-manifest.json", {"created_at": now(), "revisions": revisions, "files": files})
    print("Models prepared. Verify with: .\\run.ps1 doctor --verify-hashes --load-models", flush=True)


if __name__ == "__main__":
    main()
