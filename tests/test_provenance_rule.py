"""Answer-provenance rule: the place the task names is where the answer comes from. Offline, zero tokens.

**Root cause**: several near misses reached the right document but read the value from a similar table
next to it: one read a "Contractual Obligations" table instead of the note the task named; another
computed 100 minus "Percent Flights Late" instead of the on-time figure the page itself reports; a third
left a 10-K405 out of "10-K" filings. All decisions about where to read, not control problems.

Three things to guard: 1. with the switch off, the prompt is byte-identical to before; 2. never promise a
missing action (no mention of find_text while find_text is off); 3. **off by default and off in the
shipped config.json**: a small trial saved no task and turned one wrong answer into an empty one
(pushed onto a harder path), so it stays off until an A/B comparison says otherwise.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

import prompts  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402

REPO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")
FULL = dict(enable_back=True, enable_find_text=True, enable_read_text=True, enable_fill=True)


class TestPrompt(unittest.TestCase):
    def test_rule_present_when_enabled_and_absent_when_not(self):
        on = prompts.normal_action_space(**FULL, enable_provenance_rule=True)
        off = prompts.normal_action_space(**FULL, enable_provenance_rule=False)
        self.assertIn("take the value from exactly that place", on)
        self.assertIn("10-K405 is a 10-K", on)
        self.assertNotIn("exactly that place", off)
        self.assertEqual(off, prompts.normal_action_space(**FULL), "switched off, the prompt must be byte-identical to before")

    def test_rule_sits_next_to_the_named_filters_rule(self):
        on = prompts.normal_action_space(**FULL, enable_provenance_rule=True)
        self.assertLess(on.index("exactly that place"),
                        on.index("When the task names filters, options, tabs"))

    def test_does_not_promise_find_text_when_it_is_off(self):
        on = prompts.normal_action_space(enable_provenance_rule=True, enable_find_text=False)
        self.assertIn("exactly that place", on)
        self.assertNotIn("find_text", on)
        on2 = prompts.normal_action_space(enable_provenance_rule=True, enable_find_text=True)
        self.assertIn("use find_text(text='...') with the heading", on2)

    def test_other_switches_unaffected(self):
        on = prompts.normal_action_space(**FULL, enable_provenance_rule=True)
        for act in ["back()", "find_text(", "read_text(", "fill(", "finished("]:
            self.assertIn(act, on)


class TestConfig(unittest.TestCase):
    def test_default_off_and_shipped_config_off(self):
        """Without A/B evidence, a prompt change that every task reads should not ship."""
        self.assertFalse(RuntimeConfig().enable_provenance_rule)
        self.assertFalse(RuntimeConfig.load(config_path=REPO_CONFIG, env={}).enable_provenance_rule)

    def test_env_can_switch_on(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent.json", env={"WEBFOVEA_ENABLE_PROVENANCE_RULE": "1"})
        self.assertTrue(cfg.enable_provenance_rule)


if __name__ == "__main__":
    unittest.main()
