"""Trailing junk after an action must not eat the answer. Offline, zero tokens.

In one run the model followed every action with "\\n<br>". click/fill still worked (parsing only looks
at the prefix), but finished(content='...')\\n<br> failed ast parsing, the fallback regex required a
trailing ")", and agent_answer came out empty: **a correctly answered task scored zero at extraction**.
Both layers are fixed: the parser cuts everything after the last ")", and extraction is one notch more lenient.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

import agent as agent_mod  # noqa: E402

ANSWER = "According to USGS, there were 7 in January 2025:\\n1. M6.0 (Taiwan)\\n2. M7.1 (Tibet)"


def _extract(actions):
    # main.py imports heavy modules such as cv2 at the top; the test only takes that one function
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "agent", "main.py"), encoding="utf-8").read()
    ns = {}
    start = src.index("def extract_agent_answer")
    end = src.index("\ndef ", start + 1)
    exec("import ast, re\n" + src[start:end], ns)
    return ns["extract_agent_answer"](actions)


class TestExtractor(unittest.TestCase):
    def test_trailing_br_no_longer_loses_the_answer(self):
        self.assertEqual(_extract([f"finished(content='{ANSWER}')\n<br>"]), ANSWER)

    def test_clean_action_unchanged(self):
        self.assertEqual(_extract(["finished(content='7')"]), "7")
        self.assertEqual(_extract(["finished(content='')"]), "")

    def test_parentheses_and_quotes_inside_answer_survive(self):
        a = "finished(content='A004 (B002) and \"x\"')\n<br>"
        self.assertEqual(_extract([a]), 'A004 (B002) and "x"')


class TestParser(unittest.TestCase):
    def test_parser_strips_the_tail(self):
        text = "Thought: done.\nAction: finished(content='7 (seven)')\n<br>"
        out = agent_mod.parse_action_to_structure_output(text, 1000, 810, 1440, "claude",
                                                         screen_width=1920, screen_height=1080)
        self.assertEqual(out["action"], "finished(content='7 (seven)')")

    def test_parser_strips_tail_after_click_too(self):
        text = "Thought: click.\nAction: click(start_box='(720,405)')\n<br>"
        out = agent_mod.parse_action_to_structure_output(text, 1000, 810, 1440, "claude",
                                                         screen_width=1920, screen_height=1080)
        self.assertEqual(out["action"], "click(start_box='(960,540)')")


if __name__ == "__main__":
    unittest.main()
