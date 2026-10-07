"""read_text(): read text from the rendered page to **verify** a value read off the screenshot. Offline, zero tokens.

It verifies; it does not replace the screenshot. There are two hard reasons, and each has a test here:

  1. On sites with anti-scraping fonts the DOM holds garbage (the page shows 1098.24, the DOM holds
     U+18702...), and only the screenshot shows the true value -> looks_obfuscated must catch it
  2. The final screenshot is the evidence -> read_text does not move the view, and its feedback
     must remind the model to confirm the answer is still on screen before finishing
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

import prompts  # noqa: E402
import web_controller  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402

REPO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")


class TestObfuscationGuard(unittest.TestCase):
    """Anti-scraping-font criterion: deterministic, not heuristic."""

    def test_sanitizes_per_character_not_whole_segment(self):
        """The most important test in this file; it guards against a trap we actually fell into.

        A real 225-character line had only 7 obfuscated characters = **3.5%**, so any reasonable
        whole-passage threshold lets it through, and the obfuscated part was exactly the value to read.
        Only per-character replacement keeps the context (title / author / date) and still marks the untrustworthy positions.
        """
        # Close to reality: a listing line with a long blurb and only 7 obfuscated characters
        row = ("Red Heart Patrols the Heavens  Qing Heyi  Xianxia - Classical  Completed  "
               "(The print edition is now on sale everywhere.) In the ancient era the demon clans vanished; "
               "in the near-ancient era the dragons disappeared. The age when the gods flourished has passed like smoke, "
               "and the age of flying swords has sunk at last. What happened to this world? Who will listen? "
               "𘜂𘜃𘜉𘜄𘜆𘛾𘜅 words ·2026-08-26")
        clean, bad = web_controller.sanitize_page_text(row)
        self.assertEqual(bad, 7)
        self.assertIn("??????? words", clean)
        self.assertIn("Red Heart Patrols the Heavens", clean, "the context must not be thrown away with it")
        self.assertIn("2026-08-26", clean)
        # Counter-check: a whole-passage threshold **does** let this line through (7 of ~250 < 5%)
        ratio = bad / len([c for c in row if not c.isspace()])
        self.assertLess(ratio, 0.05, "the data must reproduce the real dilution ratio")
        self.assertFalse(web_controller.looks_obfuscated(row),
                         "which is exactly why a whole-passage threshold cannot be used: it lets realistic data through")

    def test_sanitize_leaves_normal_text_untouched(self):
        for ok in ["1098.24 words", "Angle of Attack 31.0 deg", "£27.10"]:
            clean, bad = web_controller.sanitize_page_text(ok)
            self.assertEqual((clean, bad), (ok, 0))

    def test_catches_a_fully_obfuscated_value(self):
        """When a passage is mostly garbage, looks_obfuscated still holds."""
        self.assertTrue(web_controller.looks_obfuscated("𘜂𘜃𘜉𘜄𘜆𘛾𘜅 words"))

    def test_does_not_fire_on_normal_text(self):
        """Normal text has no unnamed characters (0.0% on the data sites we checked). The CJK strings are
        there on purpose: many target sites are Chinese, and their text must never be flagged."""
        for ok in ["1098.24 words", "Angle of Attack 31.0 deg", "考研 求职 入伍",
                   "Near and Far, Geek Rating 7.226", "2024-10-01", "£27.10",
                   "第一名：复旦大学"]:
            with self.subTest(text=ok):
                self.assertFalse(web_controller.looks_obfuscated(ok))

    def test_empty_is_not_obfuscated(self):
        for s in ["", "   ", "\n\t"]:
            self.assertFalse(web_controller.looks_obfuscated(s))

    def test_emoji_and_rare_but_assigned_chars_are_fine(self):
        """Rare but assigned characters, CJK and emoji have Unicode names and must not be flagged."""
        self.assertFalse(web_controller.looks_obfuscated("营收 📈 ¥1,234 亿 ½ ℃"))


class TestThreeLayerConsistency(unittest.TestCase):
    def test_dispatch_parser_maps_read_text(self):
        """The executor dispatches on parse_action()'s name, not parse_action_type()'s."""
        got = web_controller.parse_action("read_text(near='Total revenue')")
        self.assertEqual(got["name"], "ReadText")
        self.assertEqual(got["value"], "Total revenue")

    def test_action_type_helper_also_knows_it(self):
        self.assertEqual(web_controller.parse_action_type("read_text(near='x')"), "ReadText")

    def test_positional_form_tolerated(self):
        self.assertEqual(web_controller.parse_action("read_text('Peru')")["value"], "Peru")

    def test_read_text_and_find_text_do_not_swallow_each_other(self):
        """Both contain `_text(`; in the wrong parse order one swallows the other. The regression this file must guard most."""
        self.assertEqual(web_controller.parse_action("find_text(text='a')")["name"], "FindText")
        self.assertEqual(web_controller.parse_action("read_text(near='a')")["name"], "ReadText")

    def test_advertised_only_when_enabled(self):
        self.assertIn("read_text(", prompts.normal_action_space(enable_read_text=True))
        self.assertNotIn("read_text(", prompts.normal_action_space(enable_read_text=False))

    def test_all_three_switches_coexist(self):
        space = prompts.normal_action_space(True, True, True)
        for a in ["back()", "find_text(", "read_text(", "click(", "select(", "finished("]:
            self.assertIn(a, space)
        self.assertNotIn("goto(", space)


class TestPromptDiscipline(unittest.TestCase):
    """Two easily overlooked hints that decide success or failure."""

    def test_tells_the_model_to_distrust_garbled_output(self):
        c = prompts.READ_TEXT_CONSTRAINT_LINE.lower()
        self.assertIn("scramble", c)
        self.assertIn("screenshot", c)

    def test_tells_the_model_the_view_did_not_move(self):
        """The final screenshot is the evidence: after verifying, confirm the answer is still on screen before finishing."""
        self.assertIn("does NOT move", prompts.READ_TEXT_CONSTRAINT_LINE)
        self.assertIn("last screenshot", prompts.READ_TEXT_CONSTRAINT_LINE)

    def test_trigger_is_a_rule_not_self_reported_confidence(self):
        """The trigger must be a rule such as "verify whenever the answer is a specific value",
        not the model's own "I am not sure": it is most confident exactly when it is most wrong."""
        c = prompts.READ_TEXT_CONSTRAINT_LINE
        self.assertIn("Do not skip this because you feel sure", c)


class TestBudgetAndConfig(unittest.TestCase):
    def test_enabled_in_the_shipped_config(self):
        import json
        raw = json.load(open(REPO_CONFIG, encoding="utf-8"))["agent"]
        self.assertTrue(raw.get("enable_read_text"),
                        "config.json must state it explicitly; the shipped config should explain itself, not rely on code defaults")

    def test_budget_is_tighter_than_find_text(self):
        """Every call pushes text into the context, so it needs a tighter cap than find_text."""
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertLessEqual(cfg.max_read_text_per_task, cfg.max_find_text_per_task)

    def test_output_is_capped(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertLessEqual(cfg.read_text_max_chars, 2000)

    def test_counter_clears_between_tasks(self):
        web_controller._READ_TEXT_COUNT[0] = 4
        web_controller.reset_stall_counters()
        self.assertEqual(web_controller.read_text_count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
