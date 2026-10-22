"""Loopback-only production UI. The inference workers remain fully offline."""
from __future__ import annotations

import copy
import hmac
import json
import mimetypes
import re
import secrets
import threading
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .config import load_book_config
from .pipeline import run
from .preparation import (confirm, preview, record_listening, recut_reference, replan,
                          require_confirmation, resource_estimate, update_pronunciations, update_settings)
from .project import approve, load, retry
from .report import summarize, write_report
from .runtime import StageRunner
from .storage import FileLock, contained, read_json
from .text import read_book


class Studio:
    def __init__(self, root: Path, config_path: Path | None = None):
        self.root = root.resolve()
        self.cfg = load_book_config(self.root, config_path)
        self.token = secrets.token_urlsafe(32)
        self.guard = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.task = {"active": False, "action": None, "status": "idle", "error": None}

    def snapshot(self):
        project = load(self.root)
        with self.guard:
            cfg, task = copy.deepcopy(self.cfg), copy.deepcopy(self.task)
        listening = read_json(self.root / "listening.json") if (self.root / "listening.json").is_file() else None
        if listening:
            listening["current"] = (listening["reference_sha256"] == project["reference_sha256"]
                                    and listening["profile"] == project["profile"] and listening["exports"] == project["exports"])
            if not listening["current"]:
                listening.update(accepted=False, status="outdated_audio")
        return {"summary": summarize(project), "chapters": project["chapters"],
                "segments": [{key: segment.get(key) for key in
                              ("id", "chapter_id", "text", "spoken_text", "state", "audio", "audio_sha256", "check", "error", "total_attempts")}
                             for segment in project["segments"]],
                "exports": project["exports"], "rules": read_json(self.root / "pronunciations.json"),
                "settings": {"speed": cfg["speed"], "gap_ms": cfg["gap_ms"], "scene_gap_ms": cfg["scene_gap_ms"],
                             "mp3_bitrate": cfg["export"]["mp3_bitrate"]},
                "reference": project["reference_info"], "task": task, "listening": listening,
                "estimate": resource_estimate(self.root, project, cfg),
                "production_started": bool(project["exports"] or any(r.get("mode") != "preview" for r in project["runs"]))}

    def action(self, data: dict):
        if not isinstance(data, dict) or not isinstance(data.get("action"), str):
            raise ValueError("操作格式错误。")
        action = data["action"]
        with self.guard:
            if action == "stop":
                self.cancel.set()
                return {"message": "正在暂停，已完成的片段会保留。"}
            if self.task["active"]:
                raise RuntimeError("任务运行中；请先暂停，再编辑或纠错。")
            cfg = copy.deepcopy(self.cfg)
            if action in {"run", "preview"}:
                if action == "run":
                    require_confirmation(self.root)
                if not resource_estimate(self.root, load(self.root), cfg)["disk_sufficient"]:
                    raise ValueError("可用磁盘低于制作估算，请先释放空间。")
                self.cancel.clear()
                self.task = {"active": True, "action": action, "status": "running", "error": None}
                self.thread = threading.Thread(target=self._work, args=(action, cfg), daemon=True)
                self.thread.start()
                return {"message": "试读已开始。" if action == "preview" else "制作已开始。"}
            if action == "confirm":
                result = confirm(self.root, chapters=data.get("chapters") is True,
                                 voice=data.get("voice") is True, note=str(data.get("note", "已在本地面板检查并确认")))
            elif action == "plan":
                starts = data.get("starts")
                if starts is not None and not isinstance(starts, list):
                    raise ValueError("章节行号必须为列表。")
                pattern = data.get("pattern")
                if pattern is not None and not isinstance(pattern, str):
                    raise ValueError("章节规则必须为字符串。")
                replan(self.root, cfg, pattern=pattern, starts=starts)
                result = {"message": "章节已更新，请重新确认目录与试读。"}
            elif action == "reference":
                result = recut_reference(self.root, cfg, float(data["start"]))
            elif action == "pronunciations":
                result = update_pronunciations(self.root, cfg, data.get("rules"))
            elif action == "settings":
                self.cfg = update_settings(self.root, cfg, data.get("settings"))
                result = {"message": "本书设置已保存。停顿和码率在下次导出时生效。"}
            elif action == "listening":
                result = record_listening(self.root, data.get("minutes"), data.get("checks", {}), str(data.get("note", "")))
            elif action in {"approve", "retry"}:
                ids = data.get("segments")
                if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
                    raise ValueError("请选择有效片段。")
                with FileLock(self.root / ".project.lock"):
                    if action == "approve":
                        approve(self.root, ids, str(data.get("note", "已复听，音频正确")))
                    else:
                        retry(self.root, ids)
                    write_report(self.root, load(self.root))
                result = {"message": "已标记，点击继续制作即可校对或导出。"}
            elif action == "report":
                with FileLock(self.root / ".project.lock"):
                    result = write_report(self.root, load(self.root))
            else:
                raise ValueError("未知操作。")
            return result

    def _work(self, action, cfg):
        try:
            runner = StageRunner(cfg, self.cancel)
            if action == "preview":
                preview(self.root, cfg, runner)
            else:
                run(self.root, cfg, runner)
            status, error = "complete", None
        except KeyboardInterrupt:
            status, error = "paused", None
        except Exception as exc:
            status, error = "failed", str(exc)
        with self.guard:
            self.task.update(active=False, status=status, error=error)


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, studio: Studio, port=0):
        self.studio = studio
        super().__init__(("127.0.0.1", port), StudioHandler)
        # Browser cookies are shared across ports on the same host. Each local
        # server must read only its own cookie for media authentication.
        self.cookie_name = f"shuoshu_{self.server_port}"
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.url = f"{self.origin}/session/{studio.token}/"


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "ShuoshuLocal/1.0"

    def log_message(self, *args):
        pass  # Session tokens and manuscript paths do not enter access logs.

    def _headers(self, code, content_type, size=None, **headers):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if size is not None:
            self.send_header("Content-Length", str(size))
        for key, value in headers.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()

    def _json(self, value, code=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(code, "application/json; charset=utf-8", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _trusted_host(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def _authenticated(self):
        if not self._trusted_host():
            return False
        token = self.headers.get("X-Shuoshu-Token", "")
        if not token:
            try:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
                name = self.server.cookie_name
                token = cookie[name].value if name in cookie else ""
            except Exception:
                return False
        return hmac.compare_digest(token, self.server.studio.token)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        studio = self.server.studio
        path = unquote(urlsplit(self.path).path)
        if path == f"/session/{studio.token}/" and self._trusted_host():
            body = (Path(__file__).parent / "static/studio.html").read_text(encoding="utf-8").replace("__SESSION_TOKEN__", studio.token).encode("utf-8")
            self._headers(200, "text/html; charset=utf-8", len(body),
                          Set_Cookie=f"{self.server.cookie_name}={studio.token}; HttpOnly; SameSite=Strict; Path=/")
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        if not self._authenticated():
            self._json({"error": "请使用控制台打印的本地面板地址打开。"}, 403)
            return
        try:
            if path == "/api/state":
                self._json(studio.snapshot())
            elif path == "/api/source":
                project = load(studio.root)
                source, _ = read_book(studio.root / "input/source.txt", project["encoding"])
                cursor, starts = 0, []
                lines = source.splitlines()
                for chapter in project["chapters"]:
                    for index in range(cursor, len(lines)):
                        if lines[index].strip() == chapter["title"]:
                            starts.append(index + 1)
                            cursor = index + 1
                            break
                self._json({"text": source, "starts": project.get("chapter_starts") if project.get("chapter_starts") is not None else starts})
            elif path.startswith("/media/"):
                relative = path[len("/media/"):]
                project = load(studio.root)
                allowed = {"input/reference.wav", "review.html"}
                allowed.update(s["audio"] for s in project["segments"] if s.get("audio"))
                sample = project.get("preflight", {}).get("preview")
                if sample:
                    allowed.add(sample["audio"])
                for output in project["exports"].values():
                    allowed.update("exports/" + output[key] for key in ("mp3", "wav"))
                if relative not in allowed:
                    self._json({"error": "文件不在当前书籍的可播放列表。"}, 404)
                    return
                self._file(contained(studio.root, relative))
            else:
                self._json({"error": "找不到此页面。"}, 404)
        except (OSError, ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)

    def _file(self, path):
        if not path.is_file():
            self._json({"error": "文件尚未生成。"}, 404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match or not any(match.groups()):
                self._headers(416, "application/octet-stream", 0, Content_Range=f"bytes */{size}")
                return
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
            else:
                start = max(0, size - int(match[2]))
            if start > end or start >= size:
                self._headers(416, "application/octet-stream", 0, Content_Range=f"bytes */{size}")
                return
        headers = {"Accept_Ranges": "bytes"}
        if range_header:
            headers["Content_Range"] = f"bytes {start}-{end}/{size}"
        self._headers(206 if range_header else 200, mimetypes.guess_type(path.name)[0] or "application/octet-stream", end - start + 1, **headers)
        if self.command == "HEAD":
            return
        try:
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = source.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        if (not self._authenticated() or self.headers.get("Origin") not in (None, self.server.origin)
                or self.headers.get("X-Shuoshu-Token") != self.server.studio.token):
            self._json({"error": "请求来源或会话无效。"}, 403)
            return
        if self.path != "/api/action" or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self._json({"error": "需要 JSON 操作请求。"}, 400)
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 2 * 1024 * 1024:
                raise ValueError("请求大小无效。")
            data = json.loads(self.rfile.read(size))
            self._json(self.server.studio.action(data))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self._json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, 409)


def serve(root: Path, config_path: Path | None = None, port=0, open_browser=True):
    studio = Studio(root, config_path)
    server = StudioServer(studio, port)
    print(f"本地制作面板：{server.url}", flush=True)
    print("保持此窗口开启；按 Ctrl+C 退出面板并暂停任务。", flush=True)
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        studio.cancel.set()
        if studio.thread:
            studio.thread.join(timeout=30)
        server.server_close()
