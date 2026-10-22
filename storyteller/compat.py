"""Small runtime adapters for native libraries on Windows Unicode paths."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


def load_native_file(factory, filename):
    """Kaldi's narrow Windows fopen cannot open the Chinese workspace path.

    FST basenames are ASCII. Load synchronously relative to their directory,
    restoring the worker's directory even if the native constructor fails.
    This is used only during single-threaded text-frontend initialization.
    """
    path = Path(filename).absolute()
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        return factory(path.name)
    finally:
        os.chdir(previous)


def prepare_wetext():
    if os.name != "nt":
        return
    import kaldifst
    original = kaldifst.TextNormalizer
    def unicode_loader(filename):
        return load_native_file(original, filename)
    kaldifst.TextNormalizer = unicode_loader
    try:
        import wetext  # noqa: F401 — eagerly load its bundled FST files.
    finally:
        kaldifst.TextNormalizer = original


@contextmanager
def unicode_sentencepiece():
    """Read model bytes in Python, avoiding SentencePiece's narrow fopen."""
    if os.name != "nt":
        yield
        return
    import sentencepiece
    processor = sentencepiece.SentencePieceProcessor
    original = processor.LoadFromFile
    def load_bytes(instance, filename):
        return instance.LoadFromSerializedProto(Path(filename).read_bytes())
    processor.LoadFromFile = load_bytes
    try:
        yield
    finally:
        processor.LoadFromFile = original
