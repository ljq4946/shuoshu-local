"""Pinned Windows CUDA wheel from the official PyTorch index / upstream lock."""
from pathlib import Path
from ranged_download import download

ROOT = Path(__file__).resolve().parents[2]
NAME = 'torch-2.8.0+cu128-cp311-cp311-win_amd64.whl'
URL = 'https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl'
SHA256 = '34c55443aafd31046a7963b63d30bc3b628ee4a704f826796c865fdfd05bb596'

if __name__ == '__main__':
    download(URL, ROOT / '.runtime/downloads' / NAME, SHA256)
