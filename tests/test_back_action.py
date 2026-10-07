"""back(): the action-space switch, three-layer consistency, budget and per-task reset. Offline, zero tokens.

On by default. The switch remains as a kill switch: if something goes wrong, turn it
off in the configuration without touching code.

The switched-off behaviour still has to be tested: the action line must not appear in the
action space, and the executor must reject a back() the model writes from training memory.
Asking alone does not stop it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

import prompts  # noqa: E402
import web_controller  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402


class TestSwitch(unittest.TestCase):
    """Both directions of the switch must hold."""

    def test_enabled_by_default_after_the_ruling(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertTrue(cfg.enable_back, "should be on by default")

    def test_can_be_switched_off_as_an_escape_hatch(self):
        """If something goes wrong, the configuration can turn it off without touching code."""
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_ENABLE_BACK": "false"})
        self.assertFalse(cfg.enable_back)

    def test_action_space_omits_back_when_disabled(self):
        space = prompts.normal_action_space(enable_back=False)
        self.assertNotIn("back()", space)

    def test_action_space_advertises_back_when_enabled(self):
        space = prompts.normal_action_space(enable_back=True)
        self.assertIn("back()", space)
        self.assertIn("previous page", space.lower())

    def test_enabling_back_does_not_disturb_the_rest_of_the_space(self):
        """Adding a line must not break other actions: the action space is the model's only list of abilities."""
        off = prompts.normal_action_space(False)
        on = prompts.normal_action_space(True)
        for act in ["click(", "type(", "scroll(", "hover(", "select(",
                    "reload(", "wait()", "finished("]:
            self.assertIn(act, off)
            self.assertIn(act, on)
        self.assertNotIn("goto(", on, "goto was removed and must not come back")

    def test_repo_config_actually_turns_it_on(self):
        """The shipped config.json must really have it on. A correct default with a wrong
        config file means the model never sees the action."""
        cfg = RuntimeConfig.load(config_path=os.path.join(
            os.path.dirname(__file__), "..", "config.json"), env={})
        self.assertTrue(cfg.enable_back)


class TestThreeLayerConsistency(unittest.TestCase):
    """Action space / parser / executor must line up.

    There is a history here: reload was once added to `UITARS_ACTION_SPACE` while the space
    actually in use was `UITARS_NORMAL_ACTION_SPACE`. The executor was ready, but the model was
    never told the action existed.
    A silent failure: in a run it only shows up as "the model never uses the new feature", which is very hard to trace.
    """

    def test_the_dispatch_parser_maps_back_to_the_Back_action(self):
        """This is the most important test in this file, because the bug it guards against has already happened once.

        The executor dispatches on `action["name"]` from `parse_action()`,
        **not** on the string from `parse_action_type()`; they are different functions.
        When back() was first enabled, only the latter was updated: the model dutifully issued
        6 back() calls, all of which fell into the Unknown branch as "not executed", and the executor code never ran.
        On the surface the action just "did not work"; the logs did not have a single line about it.
        """
        self.assertEqual(web_controller.parse_action("back()")["name"], "Back")

    def test_action_type_helper_also_knows_back(self):
        """The other path (agent.py uses it to count actions) must recognize it too."""
        self.assertEqual(web_controller.parse_action_type("back()"), "Back")

    def test_parser_knows_back_even_while_the_switch_is_off(self):
        """The parser must always recognize it. Otherwise it falls into Unknown, the executor's
        rejection branch is never reached, and a back() the model writes on its own goes unchecked."""
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_ENABLE_BACK": "false"})
        self.assertFalse(cfg.enable_back)
        self.assertEqual(web_controller.parse_action("back()")["name"], "Back")

    def test_back_does_not_get_confused_with_other_actions(self):
        """`in response` is a substring match; back must not steal other actions."""
        self.assertEqual(web_controller.parse_action("click(start_box='<|box_start|>(1,2)<|box_end|>')")["name"], "LeftClick")
        self.assertEqual(web_controller.parse_action("reload()")["name"], "Reload")
        self.assertEqual(web_controller.parse_action("finished(content='x')")["name"], "Finish")


class TestBudgetAndReset(unittest.TestCase):
    """Budget and per-task reset. Without a cap it becomes a new way to idle, exactly how the
    ineffective hotkeys used to burn through whole runs."""

    def test_budget_default_is_generous_enough_for_list_detail_shuttling(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertGreaterEqual(cfg.max_backs_per_task, 10,
                                "a budget this small would make back() useless on multi-record tasks")
        self.assertLess(cfg.max_backs_per_task, 40, "the budget should not approach max_steps")

    def test_counter_is_cleared_between_tasks(self):
        """Not resetting per task = later tasks inherit the budget used by earlier ones and get rejected sooner and sooner."""
        web_controller._BACK_COUNT[0] = 7
        web_controller.reset_stall_counters()
        self.assertEqual(web_controller.back_count(), 0)

    def test_counter_is_readable(self):
        web_controller._BACK_COUNT[0] = 3
        self.assertEqual(web_controller.back_count(), 3)
        web_controller.reset_stall_counters()


if __name__ == "__main__":
    unittest.main(verbosity=2)
