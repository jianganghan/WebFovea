"""Cloudflare "Verify you are human" checkbox: where does the fixed coordinate mapping actually click (no model, no network)?

Background: this was the only task in a development set that was really blocked at the entrance.
It burned all 40 steps without an answer, which looked like "cannot pass the human check". But the
trajectory shows the model got it right at step 1:

    Thought: I need to click Cloudflare's human-verification checkbox to pass the security check.
    Action:  click(start_box='(402,252)')

**It knew where to click; the harness executed the click somewhere else.** The coordinate mapping
of the time used the model's point in 1440x810 space as a 1920x1080 page coordinate, so the click
landed 1/4 of the way off towards the top left.

This test feeds that step's **real model output** to the fixed parser and asserts that the result
lands in the checkbox. The checkbox position was measured on the screenshot saved from that step
(1920x1080 original, dark outline pixels x[521,546] y[324,348]), so these are not made-up numbers
but the physical facts of that frame.

The invariant it guards: **the same model output clicked empty space before the fix and the checkbox after.**
If the coordinate mapping ever regresses, this test turns red, far cheaper than re-running the real task.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

from agent import parse_action_to_structure_output  # noqa: E402
from web_controller import parse_action  # noqa: E402

# ── Physical facts of that frame (measured on the 1920x1080 screenshot) ─────
CHECKBOX = dict(x0=521, x1=546, y0=324, y1=348)   # pixel range of the checkbox square
SENT_W, SENT_H = 1440, 810                        # size actually sent to the model after the fix
PAGE_W, PAGE_H = 1920, 1080                       # real viewport
# The model's step-1 output (thought translated, action verbatim)
STEP1 = (
    "Thought: I need to click Cloudflare's human-verification checkbox to pass the security check.\n"
    "Action: click(start_box='<|box_start|>(402,252)<|box_end|>')"
)


def _click_point(sent_w, sent_h, screen_w, screen_h):
    """Take the click point through the real chain: model text -> coordinate mapping -> action parsing -> page pixel."""
    parsed = parse_action_to_structure_output(
        STEP1, 1000, sent_h, sent_w, "qwen25vl",
        screen_width=screen_w, screen_height=screen_h,
    )
    action = parse_action(parsed["action"])          # the same step main.py takes
    x, y = action["coordinate"]
    return float(x), float(y)


def _inside(x, y):
    return CHECKBOX["x0"] <= x <= CHECKBOX["x1"] and CHECKBOX["y0"] <= y <= CHECKBOX["y1"]


class CloudflareCheckboxTest(unittest.TestCase):
    def test_lands_inside_the_checkbox_after_the_fix(self):
        x, y = _click_point(SENT_W, SENT_H, PAGE_W, PAGE_H)
        self.assertTrue(
            _inside(x, y),
            f"click point after the fix ({x:.0f},{y:.0f}) is not inside the checkbox {CHECKBOX}: the coordinate mapping is broken",
        )

    def test_missed_the_box_before_the_fix_so_the_test_can_tell_them_apart(self):
        # Old behaviour: size sent == page size, i.e. no mapping at all
        x, y = _click_point(PAGE_W, PAGE_H, PAGE_W, PAGE_H)
        self.assertFalse(
            _inside(x, y),
            "the old behaviour hits as well, so this test cannot tell before from after and tests nothing",
        )
        self.assertAlmostEqual(x, 402, delta=1)
        self.assertAlmostEqual(y, 252, delta=1)

    def test_offset_matches_the_calibration(self):
        """The ratio between the click points after and before the fix must be exactly 4/3, the factor found by calibration."""
        fx, fy = _click_point(SENT_W, SENT_H, PAGE_W, PAGE_H)
        ox, oy = _click_point(PAGE_W, PAGE_H, PAGE_W, PAGE_H)
        self.assertAlmostEqual(fx / ox, 4 / 3, delta=0.01)
        self.assertAlmostEqual(fy / oy, 4 / 3, delta=0.01)
        self.assertGreater(((fx - ox) ** 2 + (fy - oy) ** 2) ** 0.5, 150)


if __name__ == "__main__":
    unittest.main()
