"""reload(): the only remedy when a page never finished loading.

**Why it exists**: when the first paint never finished bootstrapping (endless spinner, empty content
area, a notice that scripts could not run), the model had no way out. Browser shortcuts (F5 / ctrl+r)
cannot reach Chrome's UI over CDP and are rejected, and there is no goto.
In one task the model wrote "let me try refreshing the page" in its thought and then spent 18 steps
on wait()/scroll() before giving up. In one development set, 8 of 70 tasks kept complaining that the page had not loaded.

**Why it must be capped**: given an action that looks like work but makes no progress, the model
repeats it until the steps run out, exactly how the ineffective hotkeys used to burn whole runs.
Asking does not work; the budget must be enforced in the executor.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

import prompts  # noqa: E402
import web_controller as wc  # noqa: E402
from fakes import FakeContext, FakePage  # noqa: E402


class ReloadablePage(FakePage):
    """Add reload() to FakePage, recording the call count and injected failures."""

    def __init__(self, *a, reload_error=None, **kw):
        kw.setdefault("context", FakeContext())
        super().__init__(*a, **kw)
        self.reloads = 0
        self.reload_error = reload_error
        self.url = "https://example.test/page"

    def reload(self, wait_until=None, timeout=None):
        self.reloads += 1
        if self.reload_error:
            raise RuntimeError(self.reload_error)


class TestParsing(unittest.TestCase):
    def test_parses_reload(self):
        self.assertEqual(wc.parse_action("reload()"), {"name": "Reload"})
        self.assertEqual(wc.parse_action_type("reload()"), "Reload")

    def test_reload_is_not_confused_with_url_navigation(self):
        """reload and goto are different things: one stays put, the other is navigation by URL, which is not allowed.
        Confusing them in the parser would let the forbidden action sneak in through reload."""
        self.assertEqual(wc.parse_action("goto(url='https://x.test')")["name"], "Goto")


class TestExecution(unittest.TestCase):
    def setUp(self):
        wc.reset_stall_counters()
        wc.LAST_ACTION_NOTE = ""

    def test_reloads_the_page_and_warns_that_state_is_gone(self):
        page = ReloadablePage("p")
        wc.excute_action(page, None, {"name": "Reload"})
        self.assertEqual(page.reloads, 1)
        self.assertEqual(wc.reload_count(), 1)
        # The feedback must state the cost: reloading discards filters and input already made on this page
        self.assertIn("gone", wc.take_action_note().lower())

    def test_budget_is_enforced_in_the_executor_not_the_prompt(self):
        """Writing "at most 2" in the prompt does nothing: both static constraints and runtime feedback were
        tried on this model and failed. It must be enforced in the executor."""
        page = ReloadablePage("p")
        for _ in range(wc.MAX_RELOADS_PER_TASK):
            wc.excute_action(page, None, {"name": "Reload"})
        self.assertEqual(page.reloads, wc.MAX_RELOADS_PER_TASK)

        wc.excute_action(page, None, {"name": "Reload"})          # third time
        self.assertEqual(page.reloads, wc.MAX_RELOADS_PER_TASK,   # no actual reload
                         "a reload over budget was still executed")
        note = wc.take_action_note()
        self.assertIn("REJECTED", note)
        self.assertIn("finished(content='')", note,
                      "once the budget is used up, say what to do next, or the model just switches to another action and keeps idling")

    def test_budget_resets_between_tasks(self):
        """The counter is module-level and a worker runs a dozen tasks; without a reset the second task
        could not reload even once."""
        page = ReloadablePage("p")
        for _ in range(wc.MAX_RELOADS_PER_TASK):
            wc.excute_action(page, None, {"name": "Reload"})
        wc.reset_stall_counters()                                  # a new task
        self.assertEqual(wc.reload_count(), 0)
        wc.excute_action(page, None, {"name": "Reload"})
        self.assertEqual(page.reloads, wc.MAX_RELOADS_PER_TASK + 1)

    def test_timeout_does_not_kill_the_task(self):
        """Missing domcontentloaded does not mean the page is not rendering. A timeout only reports the reason
        and must not bubble up and cost the whole task."""
        page = ReloadablePage("p", reload_error="Timeout 60000ms exceeded")
        wc.excute_action(page, None, {"name": "Reload"})           # must not raise
        note = wc.take_action_note()
        self.assertIn("still be loading", note.lower())

    def test_failed_reload_still_consumes_budget(self):
        """Failures count too: reloading N times in a row while a site is down is the hotkey idling all over again."""
        page = ReloadablePage("p", reload_error="Timeout")
        wc.excute_action(page, None, {"name": "Reload"})
        self.assertEqual(wc.reload_count(), 1)


class TestActionSpaceConsistency(unittest.TestCase):
    """Action space, parser and executor must line up.

    This test has a concrete origin: reload was first written into `UITARS_ACTION_SPACE`, while the space
    actually in use is `UITARS_NORMAL_ACTION_SPACE` (agent.py picks it with infer_mode="qwen25vl_normal").
    The executor was ready, but the model was never told the action existed: a silent failure that in a run
    only shows up as "the model never uses the new feature", very hard to trace.
    """

    ACTIVE = prompts.UITARS_NORMAL_ACTION_SPACE

    def test_every_advertised_action_can_be_parsed_and_executed(self):
        advertised = {
            "click(start_box='<|box_start|>(1,2)<|box_end|>')": "LeftClick",
            "hover(start_box='<|box_start|>(1,2)<|box_end|>')": "Hover",
            "select(start_box='<|box_start|>(1,2)<|box_end|>', option='x')": "Select",
            "reload()": "Reload",
            "wait()": "Wait",
            "finished(content='x')": "Finish",
        }
        for action_str, expected in advertised.items():
            name = action_str.split("(")[0]
            self.assertIn(f"{name}(", self.ACTIVE,
                          f"{name} can be executed but is missing from the action space in use")
            self.assertEqual(wc.parse_action(action_str)["name"], expected)

    def test_withdrawn_actions_are_not_advertised(self):
        """goto was removed and must never appear in the action space again."""
        self.assertNotIn("goto(", self.ACTIVE)

    def test_rejection_messages_never_point_at_a_withdrawn_action(self):
        """Regression: the message for a rejected hotkey once still told the model to "use goto to refresh",
        long after goto was removed, pushing the model towards an action that is always rejected."""
        page = FakePage("p", context=FakeContext())
        try:
            wc.hotkey(page, None, "ctrl+r")
        except wc.UnsupportedHotkey as e:
            msg = str(e)
        else:
            self.fail("ctrl+r should be rejected")
        self.assertNotIn("goto(", msg)
        self.assertIn("reload()", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestScrollBreakerDistinguishesReadingFromSpinning(unittest.TestCase):
    """The scroll breaker must tell "reading a long table screen by screen" from "truly idle".

    A trap we fell into: the old criterion only counted consecutive scrolls and ignored whether the page
    changed. One task was scrolling through a long table screen by screen, with new content every time;
    the breaker fired one step before it could sum up the data, and the task ended with an empty answer,
    throwing away everything it had read.

    Fix: consecutive scrolls **and** no page change; only both together count as idle.
    """

    def test_no_change_streak_accessor_exists(self):
        self.assertTrue(hasattr(wc, "no_change_streak"))

    def test_streak_starts_and_clears_at_zero(self):
        wc._NO_CHANGE_STREAK[0] = 5
        self.assertEqual(wc.no_change_streak(), 5)
        wc.reset_stall_counters()
        self.assertEqual(wc.no_change_streak(), 0)

    def test_breaker_condition_requires_both_signals(self):
        """Guard the criterion itself: the breaker branch in the source must check both the scroll count and the no-change streak."""
        src = open(os.path.join(os.path.dirname(__file__), "..",
                                "src", "agent", "main.py"), encoding="utf-8").read()
        seg = src.split("continue_scroll_cnt += 1")[1].split("curr_step += 1")[0]
        code = "\n".join(l for l in seg.splitlines() if not l.lstrip().startswith("#"))
        self.assertIn("continue_scroll_cnt >= 10", code)
        self.assertIn("no_change_streak()", code,
                      "the breaker must confirm the page really did not change, or it cuts off tasks reading long tables")
        self.assertIn("FAIL_SCROLLDOWN", code)
