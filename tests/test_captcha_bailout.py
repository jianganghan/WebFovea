"""Give up right away on a picture-grid CAPTCHA: do not spend a shared time budget on a fight that cannot be won.

Measured on one task, both arms of a comparison:
    control arm: all 40 steps burned on the CAPTCHA, "Time limit exceeded" 5 times, never reached the task page
    treatment arm: all 40 steps burned as well
Earlier runs that hit the same CAPTCHA had always submitted an empty answer within 2-3 steps;
that behaviour got lost at some point, and every encounter now cost 40 steps x ~15 s ~ 10 minutes
of a time budget **shared** by all tasks, taken from other tasks.

Why it cannot be won: the model sees all 9 images at once and judges them correctly ("top
left: black bed ✓, top middle: white bed ✓ ..."), but the action space only has "click one
point", so 4 images take 4 round trips, each a model call, and the timer runs out first.
**This is not a recognition problem; it is a structural clash between turn-taking and a
timer.**

But the rule must leave one opening: **a single-checkbox check must still be clicked**.
Cloudflare's checkbox is one such check; we calibrated coordinates for it and an offline test
pins the click point (tests/test_cloudflare_checkbox.py). A blanket rule would throw that away.
TestCheckboxCarveOut guards that opening.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

import prompts  # noqa: E402


def space():
    """Render the action space with the shipped configuration."""
    return prompts.normal_action_space(enable_back=True, enable_find_text=True,
                                       enable_read_text=True, enable_fill=True)


class TestRulePresent(unittest.TestCase):
    def test_rule_is_in_the_action_space(self):
        self.assertIn("Human-verification challenges", space())

    def test_rule_is_unconditional(self):
        """It is a **constraint**, not an ability, and hangs off no switch: it must be present in every

        combination of the action space. (Compare fill / find_text / read_text: those are new actions
        that the model must not see when switched off.)
        """
        for kw in ({}, {"enable_back": True}, {"enable_fill": True},
                   {"enable_find_text": True, "enable_read_text": True}):
            with self.subTest(**kw):
                self.assertIn("Human-verification challenges", prompts.normal_action_space(**kw))

    def test_says_what_to_do_not_just_what_not_to_do(self):
        """Only saying "don't" leaves the model stuck. It must be given the next step: submit an empty answer."""
        sp = space()
        i = sp.index("Human-verification challenges")
        rule = sp[i:i + 700]
        self.assertIn("finished(content='')", rule)


class TestCheckboxCarveOut(unittest.TestCase):
    """Single-checkbox checks must not be caught by this rule: one task depends on exactly that click."""

    def test_single_checkbox_is_explicitly_allowed(self):
        sp = space()
        i = sp.index("Human-verification challenges")
        rule = sp[i:i + 700]
        self.assertIn("single box", rule)
        self.assertIn("click it once", rule)

    def test_bailout_is_scoped_to_the_picture_grid(self):
        """The criterion must be a **visible shape** (a grid of pictures), not "feels like a CAPTCHA".

        The model cannot follow a vaguely worded rule: triggers written as rules were adopted in 5 of 6
        tasks, triggers based on the model's own confidence in 2 of 6.
        """
        sp = space()
        i = sp.index("Human-verification challenges")
        rule = sp[i:i + 700]
        self.assertIn("GRID OF PICTURES", rule)

    def test_cloudflare_checkbox_test_still_exists(self):
        """If this rule is ever widened, the checkbox calibration work is wasted. Leave a cross-reference."""
        here = os.path.dirname(__file__)
        self.assertTrue(os.path.exists(os.path.join(here, "test_cloudflare_checkbox.py")))


class TestShippedConfig(unittest.TestCase):
    """Switch states in the shipped configuration: easy to forget after a change, so let a test remember."""

    def test_fill_and_newline_submit_are_on(self):
        import json
        cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json"),
                             encoding="utf-8"))["agent"]
        self.assertTrue(cfg["enable_fill"], "paired comparison: 2 fixed, 0 broken, adopted in 7/7, 0 of 31 silent failures")
        self.assertTrue(cfg["enable_type_newline_submit"],
                        "a single \\n once cost a task 9 steps")


if __name__ == "__main__":
    unittest.main()
