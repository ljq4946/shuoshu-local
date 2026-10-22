from __future__ import annotations

import contextlib
import http.cookiejar
import json
import threading
import urllib.error
import urllib.request

from storyteller.studio import Studio, StudioServer
from tests.support import BookTestCase


class StudioSessionTests(BookTestCase):
    @contextlib.contextmanager
    def panels(self):
        servers = [StudioServer(Studio(self.book(name=name))) for name in ("first", "second")]
        threads = [threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
                   for server in servers]
        for thread in threads:
            thread.start()
        try:
            yield servers
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=3)

    def request(self, opener, url, headers=None, data=None):
        request = urllib.request.Request(url, headers=headers or {}, data=data)
        try:
            response = opener.open(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read(), response.headers

    def browser(self):
        return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def test_two_panels_in_one_browser_keep_both_audio_sessions(self):
        with self.panels() as (first, second):
            browser = self.browser()
            self.assertEqual(self.request(browser, first.url)[0], 200)
            self.assertEqual(self.request(browser, first.origin + "/media/input/reference.wav")[0], 200)
            self.assertEqual(self.request(browser, second.url)[0], 200)
            for server in (first, second, first):
                self.assertEqual(self.request(browser, server.origin + "/media/input/reference.wav")[0], 200)
                self.assertEqual(self.request(browser, server.origin + "/api/state")[0], 200)

    def test_anonymous_and_cross_session_tokens_remain_rejected(self):
        with self.panels() as (first, second):
            anonymous, browser = urllib.request.build_opener(), self.browser()
            self.request(browser, first.url)
            for url in ("/api/state", "/media/input/reference.wav"):
                self.assertEqual(self.request(anonymous, first.origin + url)[0], 403)
                self.assertEqual(self.request(browser, first.origin + url, {"X-Shuoshu-Token": second.studio.token})[0], 403)
            # A valid cookie alone cannot authorize a state-changing request.
            headers = {"Content-Type": "application/json", "Origin": first.origin}
            data = json.dumps({"action": "stop"}).encode()
            self.assertEqual(self.request(browser, first.origin + "/api/action", headers, data)[0], 403)
            headers["X-Shuoshu-Token"] = first.studio.token
            self.assertEqual(self.request(browser, first.origin + "/api/action", headers, data)[0], 200)
            headers["Origin"] = second.origin
            self.assertEqual(self.request(browser, first.origin + "/api/action", headers, data)[0], 403)

    def test_media_range_and_path_guards_work_with_both_cookies(self):
        with self.panels() as (first, second):
            browser = self.browser()
            for server in (first, second):
                self.request(browser, server.url)
            for server in (first, second):
                status, body, headers = self.request(browser, server.origin + "/media/input/reference.wav", {"Range": "bytes=0-31"})
                self.assertEqual((status, len(body)), (206, 32))
                self.assertTrue(headers["Content-Range"].startswith("bytes 0-31/"))
                self.assertEqual(self.request(browser, server.origin + "/media/../project.json")[0], 404)
