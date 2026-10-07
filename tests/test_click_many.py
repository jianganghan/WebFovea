"""click_many(): click a picture-grid CAPTCHA in one turn. Offline, zero tokens.

**Why it exists**: the agent used to give up immediately on picture-grid checks, because each
square costs one model call (10-20 s) and the challenge times out first ("time limit exceeded").
Batching brings a round down to a single call.

Lines it must hold:
  1. all four places line up (action space / parse_action_type / parse_action / executor); one missing = silent failure
  2. coordinates go through agent.py's remapping (the model sees 1440x810, the page is 1920x1080)
  3. the executor caps it at 16 points and 6 calls per task; switched off, it is rejected and the prompt is byte-identical
"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

import agent as agent_mod  # noqa: E402
import prompts  # noqa: E402
import web_controller as wc  # noqa: E402
from fakes import FakeContext, FakePage  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402

REPO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")
FULL = dict(enable_back=True, enable_find_text=True, enable_read_text=True, enable_fill=True)
ACT = "click_many(points='(100,100);(300,100);(500,300)')"


class FakeMouse:
    def __init__(self):
        self.clicks = []

    def click(self, x, y, **kw):
        self.clicks.append((x, y))


class GridPage(FakePage):
    def __init__(self):
        super().__init__(context=FakeContext())
        self.mouse = FakeMouse()
        self.frames = []


class TestAlignment(unittest.TestCase):
    def test_both_parsers_know_it(self):
        self.assertEqual(wc.parse_action_type(ACT), "ClickMany")
        got = wc.parse_action(ACT)
        self.assertEqual(got, {"name": "ClickMany", "points": [[100, 100], [300, 100], [500, 300]]})

    def test_does_not_steal_click(self):
        self.assertEqual(wc.parse_action_type("click(start_box='(1,2)')"), "LeftClick")
        self.assertEqual(wc.parse_action("click(start_box='<|box_start|>(1,2)<|box_end|>')")["name"], "LeftClick")

    def test_box_tokens_inside_points_are_tolerated(self):
        got = wc.parse_action("click_many(points='<|box_start|>(10,20)<|box_end|>;<|box_start|>(30,40)<|box_end|>')")
        self.assertEqual(got["points"], [[10, 20], [30, 40]])

    def test_empty_points_is_unknown(self):
        self.assertEqual(wc.parse_action("click_many(points='')")["name"], "Unknown")

    def test_prompt_switch(self):
        on = prompts.normal_action_space(**FULL, enable_click_many=True)
        off = prompts.normal_action_space(**FULL, enable_click_many=False)
        self.assertIn("click_many(points=", on)
        self.assertIn("issue ONE click_many", on)
        self.assertNotIn("stop and call finished(content='') right away", on)
        self.assertNotIn("click_many", off)
        self.assertEqual(off, prompts.normal_action_space(**FULL), "switched off, the prompt is byte-identical to before")
        self.assertIn("GRID OF PICTURES", on, "the rule still targets picture grids only")
        self.assertIn("single box", on, "a single checkbox is still clicked")


class TestRemap(unittest.TestCase):
    def test_every_point_is_scaled_from_sent_image_to_page(self):
        text = "Thought: three squares.\nAction: click_many(points='(720,405);(144,81);(1440,810)')"
        out = agent_mod.parse_action_to_structure_output(
            text, 1000, 810, 1440, "claude", screen_width=1920, screen_height=1080)
        self.assertEqual(out["action"], "click_many(points='(960,540);(192,108);(1920,1080)')")


class TestExecution(unittest.TestCase):
    def setUp(self):
        wc.reset_stall_counters()
        wc.LAST_ACTION_NOTE = ""

    def test_clicks_every_point_in_order_and_explains_next_step(self):
        page = GridPage()
        wc.excute_action(page, None, wc.parse_action(ACT))
        self.assertEqual(page.mouse.clicks, [(100, 100), (300, 100), (500, 300)])
        note = wc.take_action_note()
        self.assertIn("clicked 3 points", note)
        self.assertIn("Verify", note)

    def test_caps_at_16_points(self):
        pts = ";".join(f"({i * 10},{i * 10})" for i in range(25))
        page = GridPage()
        wc.excute_action(page, None, wc.parse_action(f"click_many(points='{pts}')"))
        self.assertEqual(len(page.mouse.clicks), wc.MAX_CLICK_MANY_POINTS)

    def test_per_task_budget(self):
        page = GridPage()
        for _ in range(wc.MAX_CLICK_MANY_PER_TASK):
            wc.excute_action(page, None, wc.parse_action(ACT))
        n = len(page.mouse.clicks)
        wc.excute_action(page, None, wc.parse_action(ACT))
        self.assertEqual(len(page.mouse.clicks), n, "no more clicks once over budget")
        self.assertIn("REJECTED", wc.take_action_note())
        wc.reset_stall_counters()
        wc.excute_action(page, None, wc.parse_action(ACT))
        self.assertEqual(len(page.mouse.clicks), n + 3, "reset between tasks")

    def test_rejected_when_switched_off(self):
        page = GridPage()
        with unittest.mock.patch.object(wc, "get_config", return_value=RuntimeConfig(enable_click_many=False)):
            wc.excute_action(page, None, wc.parse_action(ACT))
        self.assertEqual(page.mouse.clicks, [])
        self.assertIn("REJECTED", wc.take_action_note())


class TestBudgetAccounting(unittest.TestCase):
    def test_click_many_costs_one_step_per_point(self):
        """Steps count browser operations; the max_steps budget must be charged the same way, or a hard step limit could be exceeded unnoticed."""
        self.assertEqual(wc.step_cost(wc.parse_action(ACT)), 3)
        self.assertEqual(wc.step_cost({"name": "LeftClick", "coordinate": [1, 2]}), 1)
        pts = ";".join(f"({i},{i})" for i in range(25))
        self.assertEqual(wc.step_cost(wc.parse_action(f"click_many(points='{pts}')")), wc.MAX_CLICK_MANY_POINTS)
        self.assertEqual(wc.step_cost({"name": "ClickMany", "points": []}), 1)

    def test_main_loop_uses_the_step_cost(self):
        src = open(os.path.join(os.path.dirname(__file__), "..", "src", "agent", "main.py"), encoding="utf-8").read()
        self.assertIn("effect_step += web_controller.step_cost(action)", src)

    def test_pause_before_the_last_point_so_verify_sees_the_selection(self):
        page = GridPage()
        wc.excute_action(page, None, wc.parse_action(ACT))
        self.assertGreaterEqual(page.waited_ms, 700 + 3 * 350 + 1200)

    def test_prompt_lets_verify_ride_in_the_same_call_for_static_grids(self):
        on = prompts.normal_action_space(**FULL, enable_click_many=True)
        self.assertIn("Verify button as the LAST point", on)
        self.assertIn("click verify once there are none left", on)


class TestConfig(unittest.TestCase):
    def test_default_and_shipped_config(self):
        self.assertTrue(RuntimeConfig().enable_click_many)
        self.assertTrue(RuntimeConfig.load(config_path=REPO_CONFIG, env={}).enable_click_many)
        cfg = RuntimeConfig.load(config_path="/nonexistent.json", env={"WEBFOVEA_ENABLE_CLICK_MANY": "0"})
        self.assertFalse(cfg.enable_click_many)


if __name__ == "__main__":
    unittest.main()
