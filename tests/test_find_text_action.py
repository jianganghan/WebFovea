"""find_text(): find on page. Offline, zero tokens.

One find_text counts as one step: locating is free, only operating the browser counts, and
scrolling the match into view is that one operation.

**This action has a line that is easy to cross**: Playwright's Locator offers
`.inner_text()` / `.text_content()`, which would read the answer in one line, much easier than
a screenshot. **That would get data around the page's interface**, the same category as
navigating by URL. The action only locates and scrolls; the answer is still read from the
screenshot. One test below guards exactly this line.

Action space / parser / executor must line up, and **the parser has two functions**:
when back() was added only to the counting one, all 6 of the model's calls fell into Unknown and the executor never ran.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

import prompts  # noqa: E402
import web_controller  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402

REPO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")


class TestThreeLayerConsistency(unittest.TestCase):
    def test_the_dispatch_parser_maps_find_text_to_FindText(self):
        """The executor dispatches on parse_action()'s name, not parse_action_type()'s."""
        got = web_controller.parse_action("find_text(text='Angle of Attack')")
        self.assertEqual(got["name"], "FindText")
        self.assertEqual(got["value"], "Angle of Attack")

    def test_action_type_helper_also_knows_it(self):
        self.assertEqual(web_controller.parse_action_type("find_text(text='x')"), "FindText")

    def test_positional_form_is_tolerated(self):
        """The model may write find_text('xxx'); it must not fall into Unknown."""
        self.assertEqual(web_controller.parse_action("find_text('Peru')")["value"], "Peru")

    def test_advertised_when_enabled_and_hidden_when_not(self):
        self.assertIn("find_text(", prompts.normal_action_space(enable_find_text=True))
        self.assertNotIn("find_text(", prompts.normal_action_space(enable_find_text=False))

    def test_back_and_find_text_can_both_be_on(self):
        """The two switches must not interfere; string splicing goes wrong most easily here."""
        space = prompts.normal_action_space(enable_back=True, enable_find_text=True)
        for act in ["back()", "find_text(", "click(", "select(", "reload(", "finished("]:
            self.assertIn(act, space)
        self.assertNotIn("goto(", space)

    def test_does_not_steal_other_actions(self):
        """`in response` is a substring match; do not steal other actions."""
        self.assertEqual(web_controller.parse_action("reload()")["name"], "Reload")
        self.assertEqual(web_controller.parse_action("back()")["name"], "Back")
        self.assertEqual(
            web_controller.parse_action("click(start_box='<|box_start|>(1,2)<|box_end|>')")["name"],
            "LeftClick")


class TestComplianceLine(unittest.TestCase):
    """Locate and scroll only; never read values straight from the DOM."""

    def test_executor_never_reads_text_off_the_dom(self):
        """Guard the line that is easiest to cross: no Locator value-reading method may appear in the executor branch.

        Why assert on source: this constraint cannot be tested through behaviour (not calling something
        has no behaviour), but crossing it would break the agent's core rule. A failing test works better than a comment.
        """
        src = open(os.path.join(os.path.dirname(__file__), "..",
                                "src", "agent", "web_controller.py"), encoding="utf-8").read()
        seg = src.split('elif name == "FindText":')[1].split('elif name == "Back":')[0]
        # Strip comments before scanning: the comments name these methods **on purpose** to warn future
        # readers, which is documentation, not a call. Scan only code that runs.
        code = "\n".join(l for l in seg.splitlines() if not l.lstrip().startswith("#"))
        for forbidden in (".inner_text(", ".text_content(", ".all_text_contents(",
                          ".input_value(", ".get_attribute("):
            self.assertNotIn(forbidden, code,
                             f"{forbidden} appears in the find_text branch; that would read data around the interface")
        # Reverse check: the test really can catch something (otherwise over-stripping would go unnoticed)
        self.assertIn("scroll_into_view_if_needed", code)


class TestBudgetAndConfig(unittest.TestCase):
    def test_enabled_in_the_shipped_config(self):
        self.assertTrue(RuntimeConfig.load(config_path=REPO_CONFIG, env={}).enable_find_text)

    def test_timeout_is_short_not_playwright_default(self):
        """When the text does not exist, Playwright waits out the full timeout before raising; the 30 s default is a time sink."""
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertLessEqual(cfg.find_text_timeout_ms, 5000)

    def test_budget_exists_and_is_sane(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertGreaterEqual(cfg.max_find_text_per_task, 5)
        self.assertLess(cfg.max_find_text_per_task, 40)

    def test_counter_clears_between_tasks(self):
        web_controller._FIND_TEXT_COUNT[0] = 6
        web_controller.reset_stall_counters()
        self.assertEqual(web_controller.find_text_count(), 0)

    def test_can_be_switched_off(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_ENABLE_FIND_TEXT": "false"})
        self.assertFalse(cfg.enable_find_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
