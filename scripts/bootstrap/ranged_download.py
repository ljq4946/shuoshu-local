"""Resumable, hash-verified downloads of large, immutable public artifacts."""
from __future__ import annotations

import concurrent.futures
import hashlib
import math
import os
import re
import time
import threading
from pathlib import Path


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def download(url: str, output: Path, expected_hash: str, size: int | None = None,
             workers: int = 6, chunk_size: int = 16 * 1024 * 1024):
    import requests
    if not re.fullmatch(r'[0-9a-f]{64}', expected_hash):
        raise ValueError('A verified SHA-256 is required before downloading.')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and (size is None or output.stat().st_size == size) and sha256(output) == expected_hash:
        print(f'Already verified: {output.name}', flush=True)
        return output
    for attempt in range(4):
        try:
            with requests.get(url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'}, stream=True, timeout=(30, 60)) as probe:
                probe.raise_for_status()
                match = re.fullmatch(r'bytes 0-0/(\d+)', probe.headers.get('Content-Range', ''))
                if probe.status_code != 206 or not match:
                    raise RuntimeError('Server does not support exact byte ranges.')
                total = int(match[1])
            break
        except requests.exceptions.RequestException as exc:
            if attempt == 3 or (getattr(exc, 'response', None) is not None and 400 <= exc.response.status_code < 500):
                raise
            print(f'Retrying connection: {output.name} ({attempt + 1}/3)', flush=True)
            time.sleep(2 ** attempt)
    if size is not None and size != total:
        raise RuntimeError(f'Server size does not match the pinned artifact: {total} != {size}')
    size = total
    parts = output.parent / '.chunks' / expected_hash / str(chunk_size)
    parts.mkdir(parents=True, exist_ok=True)
    count = math.ceil(size / chunk_size)
    stopped = threading.Event()

    def part(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        length = end - start + 1
        target = parts / f'{index:06d}.part'
        if target.is_file() and target.stat().st_size == length:
            return target
        temporary = target.with_suffix('.downloading')
        for attempt in range(4):
            if stopped.is_set():
                raise concurrent.futures.CancelledError()
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Server returned a different range; refusing to assemble corrupt data.')
                    received = 0
                    with temporary.open('wb') as stream:
                        for block in response.iter_content(512 * 1024):
                            if stopped.is_set():
                                raise concurrent.futures.CancelledError()
                            received += len(block)
                            if received > length:
                                raise RuntimeError('Range response exceeds expected size.')
                            stream.write(block)
                    if received != length:
                        raise RuntimeError('Incomplete range response.')
                os.replace(temporary, target)
                return target
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 3 or stopped.is_set():
                    raise
                time.sleep(2 ** attempt)

    print(f'Downloading {output.name}: {size / 1024**3:.2f} GiB, {workers} connections', flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(part, index) for index in range(count)]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed == count or completed % max(1, count // 10) == 0:
                print(f'  {output.name}: {completed}/{count} chunks', flush=True)
    finally:
        stopped.set()
        pool.shutdown(wait=True, cancel_futures=True)
    assembled = output.with_name(output.name + '.assembling')
    try:
        with assembled.open('wb') as stream:
            for index in range(count):
                with (parts / f'{index:06d}.part').open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        stream.write(block)
        if assembled.stat().st_size != size or sha256(assembled) != expected_hash:
            # These files belong exclusively to this SHA-256 cache directory.
            for index in range(count):
                (parts / f'{index:06d}.part').unlink(missing_ok=True)
            raise RuntimeError(f'SHA-256 mismatch: {output.name}; never installed this artifact.')
        os.replace(assembled, output)
        for index in range(count):
            (parts / f'{index:06d}.part').unlink(missing_ok=True)
        print(f'Verified SHA-256: {output.name}', flush=True)
        return output
    finally:
        assembled.unlink(missing_ok=True)
