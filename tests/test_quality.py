from __future__ import annotations

import copy
import unittest

from storyteller.config import DEFAULTS, validate_config
from storyteller.quality import evaluate, normalize


class NumericQualityTests(unittest.TestCase):
    def setUp(self):
        self.qa = copy.deepcopy(DEFAULTS["qa"])

    def check(self, expected, recognized):
        return evaluate(expected, recognized, max(2, len(normalize(expected)) / 4), self.qa)

    def test_changed_numeric_structures_require_review(self):
        # Isolate the numeric gate from ordinary CER and gap thresholds.
        self.qa.update(max_cer=1, max_gap=1000)
        for expected, recognized in (
            ("3-5", "35"), ("3–5", "35"), ("3—5", "35"), ("3～5", "35"),
            ("1/2", "12"), ("1⁄2", "12"), ("1、2、3", "123"),
            ("1 2 3", "123"), ("1、2、3", "1、3、2"), ("3-5", "3、5"),
            ("1/2", "2/1"), ("-5--1", "-5-1"), ("-5", "5"),
        ):
            with self.subTest(expected=expected, recognized=recognized):
                result = self.check("数值为" + expected + "，请核对后保存记录。", "数值为" + recognized + "，请核对后保存记录。")
                self.assertFalse(result["passed"])
                self.assertIn("关键数值或正负号不一致", result["reasons"])

    def test_equivalent_numeric_readings_still_pass(self):
        for expected, recognized in (
            ("行程3-5公里。", "行程三到五公里。"),
            ("行程3~5公里。", "行程三至五公里。"),
            ("温度-5--1度。", "温度零下五到负一度。"),
            ("取总数的1/2。", "取总数的二分之一。"),
            ("数值为-1/2。", "数值为负二分之一。"),
            ("编号1、2、3。", "编号一，二，三。"),
            ("温度-5度。", "温度零下五度。"),
            ("预算-500元。", "预算负五百元。"),
            ("数额1,234,567元。", "数额一百二十三万四千五百六十七元。"),
            ("单价3.14元。", "单价三点一四元。"),
            ("利率-3.5%。", "利率负百分之三点五。"),
            ("出发时间12:30。", "出发时间十二点三十分。"),
            ("今天是2026/10/21。", "今天是二零二六年十月二十一日。"),
            ("今天是2026-10-21。", "今天是二零二六年十月二十一日。"),
        ):
            with self.subTest(expected=expected, recognized=recognized):
                self.assertEqual(normalize(expected), normalize(recognized))
                self.assertTrue(self.check(expected, recognized)["passed"])

    def test_ordinary_punctuation_remains_ignorable(self):
        for expected, recognized in (("你好，世界。", "你好世界"), ("第一章，雨夜。", "第一章雨夜")):
            self.assertTrue(self.check(expected, recognized)["passed"])


class CriticalTermTests(unittest.TestCase):
    def test_changed_critical_term_is_detected_after_normalization(self):
        for term, old, new in (("AI", "AI", ""), ("一鸣", "一鸣", "一明"), ("ＡＩ", "AI", "")):
            with self.subTest(term=term):
                qa = copy.deepcopy(DEFAULTS["qa"])
                qa["critical_terms"] = [term]
                expected = f"这次讨论围绕{old}展开，请各位认真核对每个环节并保存相关文字资料，以便后续继续阅读。"
                recognized = expected.replace(old, new)
                result = evaluate(expected, recognized, 12, qa)
                self.assertLessEqual(result["cer"], qa["max_cer"])
                self.assertFalse(result["passed"])
                self.assertIn("疑似关键字差异：" + term, result["reasons"])

    def test_equivalent_critical_term_spelling_passes(self):
        for term, expected, recognized in (("AI", "这是AI技术。", "这是ai技术。"),
                                             ("一鸣", "一鸣回家了。", "1鸣回家了。"),
                                             ("1/2", "数值为1/2。", "数值为二分之一。")):
            with self.subTest(term=term):
                qa = copy.deepcopy(DEFAULTS["qa"])
                qa["critical_terms"] = [term]
                self.assertTrue(evaluate(expected, recognized, 3, qa)["passed"])

    def test_empty_or_duplicate_normalized_critical_terms_are_rejected(self):
        for terms, message in ((["!!!"], "规范化后为空"), (["AI", "ai"], "规范化后重复"),
                               (["一鸣", "1鸣"], "规范化后重复")):
            with self.subTest(terms=terms):
                cfg = copy.deepcopy(DEFAULTS)
                cfg["qa"]["critical_terms"] = terms
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
