"""Read-only inventory and integrity checks for the audit snapshot."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from storyteller.config import load_book_config, load_config
from storyteller.storage import contained, file_hash, now, read_json, write_json
from storyteller.training_data import read_jsonl, verify_sources


def check_hash(path, expected):
    if not path.is_file():
        return {"path": str(path), "error": "missing"}
    actual = file_hash(path)
    return None if actual == expected else {"path": str(path), "error": "hash_mismatch", "actual": actual, "expected": expected}


def main():
    output = Path(__file__).resolve().parent
    source = sorted(list((ROOT / "storyteller").rglob("*.py")) + list((ROOT / "scripts/bootstrap").glob("*.py")))
    for path in source:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    protected = source + list((ROOT / "storyteller/static").glob("*.html")) + list(ROOT.glob("*.ps1")) + list((ROOT / "scripts").glob("*.ps1"))
    protected += [ROOT / "config.local.json", ROOT / "pyproject.toml", ROOT / "README.md"]
    cfg = load_config()
    result = {"checked_at_utc": now(), "root": str(ROOT), "python_syntax_files": len(source),
              "python_syntax": "passed", "tests_directory_present": (ROOT / "tests").is_dir(),
              "validation_directory_present": (ROOT / "validation").is_dir(), "git_repository_present": (ROOT / ".git").exists(),
              "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in protected},
              "production_adapter": cfg["tts_adapter"], "books": []}
    for project_path in sorted((ROOT / "books").glob("*/project.json")):
        book, project = project_path.parent, read_json(project_path)
        checked, errors, with_audio = 0, [], [s for s in project["segments"] if s.get("audio")]
        for segment in with_audio:
            if segment.get("audio_sha256"):
                checked += 1
                error = check_hash(contained(book, segment["audio"]), segment["audio_sha256"])
                if error:
                    errors.append(error)
        inputs = []
        for key, relative in (("source_sha256", "input/source.txt"), ("reference_sha256", "input/reference.wav")):
            error = check_hash(book / relative, project[key])
            if error:
                inputs.append(error)
        entry = {"path": str(book), "project_sha256": file_hash(project_path),
                 "chapters": len(project["chapters"]), "segments": len(project["segments"]),
                 "states": dict(Counter(s["state"] for s in project["segments"])),
                 "pipeline_exports": len(project["exports"]), "segments_referencing_audio": len(with_audio),
                 "segment_audio_hashes_checked": checked, "segment_audio_hash_errors": errors, "input_hash_errors": inputs,
                 "profile": project["profile"], "book_adapter": load_book_config(book)["tts_adapter"],
                 "listening_record_exists": (book / "listening.json").is_file()}
        manifest_path = book / "有声书成品/book_manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            pack_errors, pack_checked = [], 0
            for track in manifest["tracks"]:
                for key in ("wav", "mp3"):
                    pack_checked += 1
                    error = check_hash(contained(manifest_path.parent, track[key]), track["sha256"][key])
                    if error:
                        pack_errors.append(error)
            entry["separate_audio_package"] = {"expected_tracks": manifest["expected_tracks"],
                                               "published_tracks": manifest["published_tracks"],
                                               "duration_seconds": manifest["duration"],
                                               "review_flagged_segments": sum(t.get("review_segments", 0) for t in manifest["tracks"]),
                                               "file_hashes_checked": pack_checked, "hash_errors": pack_errors,
                                               "manifest_sha256": file_hash(manifest_path)}
        result["books"].append(entry)
    dataset = ROOT / "training/qiong1_20261022"
    report, summary = read_json(dataset / "dataset_report.json"), read_json(dataset / "runs/lora_rank16/summary.json")
    splits = {name: read_jsonl(dataset / (name + ".jsonl")) for name in ("train", "validation", "test")}
    manifest_errors = [name for name, expected in report["manifest_sha256"].items()
                       if file_hash(dataset / (name + ".jsonl")) != expected]
    index = read_json(dataset / "features/index.json")
    feature_errors, features_checked = [], 0
    for rows in splits.values():
        for row in rows:
            features_checked += 1
            record = index.get(row["id"], {})
            error = check_hash(contained(dataset / "features", row["id"] + ".pt"), record.get("sha256"))
            if error:
                feature_errors.append(error)
    leaks = []
    for i, first in enumerate(splits):
        for second in list(splits)[i + 1:]:
            for key in ("id", "text_key", "pcm_sha256"):
                overlap = {r[key] for r in splits[first]} & {r[key] for r in splits[second]}
                if overlap:
                    leaks.append({"first": first, "second": second, "key": key, "count": len(overlap)})
    checkpoint_error = check_hash(Path(summary["best_checkpoint"]), summary["best_checkpoint_sha256"])
    result["training"] = {"source_check": verify_sources(dataset), "splits": {k: len(v) for k, v in splits.items()},
                          "split_manifest_hash_errors": manifest_errors, "cross_split_exact_leaks": leaks,
                          "feature_files_checked": features_checked, "feature_hash_errors": feature_errors,
                          "best_checkpoint_hash_error": checkpoint_error, "summary_status": summary["status"],
                          "human_listening": summary["human_listening"], "chinese_improvement": summary["chinese_improvement"]}
    result["model_manifests"] = {key: {"manifest_sha256": file_hash(Path(cfg[key]) / "asset-manifest.json"),
                                       "listed_files": len(read_json(Path(cfg[key]) / "asset-manifest.json")["files"])}
                                  for key in ("index_model", "asr_model", "asr_vad_model")}
    write_json(output / "artifact-integrity.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"source_sha256"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
