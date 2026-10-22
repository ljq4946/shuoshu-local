from __future__ import annotations

import re
import unittest

from storyteller.text import is_heading, split_chapters


class ChapterDetectionTests(unittest.TestCase):
    def test_chapter_like_body_lines_do_not_create_extra_chapters(self):
        for body in ("第一回见面时……", "第一回见面时我们都还年轻", "第一回见面时...",
                     "第一回见面时我们都还年轻。", "第一节课刚开始", "第二部电影还没看完",
                     "第三篇文章读完了", "第二集节目刚开始"):
            with self.subTest(body=body):
                source = f"第一章 开始\n这是正文。\n{body}\n后来发生了很多事情。\n第二章 结束\n这是结尾。"
                chapters = split_chapters(source, "样例")
                self.assertEqual([c["title"] for c in chapters], ["第一章 开始", "第二章 结束"])
                compact = lambda text: re.sub(r"\s+", "", text)
                self.assertEqual(compact("".join(c["text"] for c in chapters)), compact(source))

    def test_standard_and_compact_chapter_titles_are_preserved(self):
        for title in ("第一章雨夜", "第一卷风起", "第十二章", "第12章：重逢", "第一回 初见",
                      "第二回：重逢", "第三节 窗下", "第一部 起点", "第2集：雨夜", "第三篇 远行",
                      "Chapter 1 Rain", "序章", "楔子：往事", "尾声", "后记"):
            with self.subTest(title=title):
                self.assertTrue(is_heading(title))
                self.assertEqual(split_chapters(title + "\n这是一段正文。", "样例")[0]["title"], title)

    def test_custom_pattern_preserves_compact_classical_chapter_titles(self):
        source = "第一回甄士隐梦幻识通灵\n这是一段正文。\n第二回贾夫人仙逝扬州城\n这是另一段正文。"
        chapters = split_chapters(source, "样例", pattern=r"第[一二三四五六七八九十]+回.*")
        self.assertEqual([c["title"] for c in chapters], ["第一回甄士隐梦幻识通灵", "第二回贾夫人仙逝扬州城"])

    def test_explicit_start_lines_override_automatic_ambiguity_checks(self):
        source = "开篇文字。\n第一回初见\n第一回见面时……\n第二回再会\n结束。"
        chapters = split_chapters(source, "样例", starts=[2, 4])
        self.assertEqual([c["title"] for c in chapters], ["样例", "第一回初见", "第二回再会"])
        self.assertIn("第一回见面时……", chapters[1]["text"])
