"""Strip chat-template end tokens the model makes up (`<|eot_id|>` / `<|end_of_turn|>` ...). Offline, zero tokens.

A scan of all past runs found **15 of 305 episodes (4.9%) contaminated by such a token, 14 of
them scored zero**. Each of its two ways of doing damage has a test here:

  1. Direct damage: the parser swallowed the `<|eot_id|>` line after `Action: type(content='Peru')`
     into content, so the agent **typed the end token into the input box as text**. One episode typed
     `2022 June<|end_of_turn|>` into a filter box and the search kept answering "No Results"; the model
     diagnosed this in its own thought and still could not get around it, running out of all 40 steps.

  2. Fatal damage: the reply goes into `history_responses` unchanged, so on the next step the model
     sees itself using the token and keeps doing it. Every one of the 15 episodes had it from step 0
     onwards. So stripping must happen in `_truncate_to_first_action` (the only way into the history),
     not in the executor.

The easiest place to get wrong: `<|box_start|>` / `<|box_end|>` are **part of the coordinate syntax**;
   stripping them by mistake would leave every click without a target. TestBoxTokensSurvive pins this down.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

from agent import (  # noqa: E402
    _truncate_to_first_action as truncate,
    parse_action_to_structure_output as parse,
)

# Same call as in agent.predict(); this file only cares about the parse result, so any valid size will do
PARSE_ARGS = (1000, 1080, 1920, "qwen25vl", 16384 * 28 * 28, 100 * 28 * 28)


def action_of(raw):
    return parse(truncate(raw), *PARSE_ARGS)["action"]


class TestDamageChannelOne(unittest.TestCase):
    """1. The path into the input box. The data is the model's real output from past runs, not made up."""

    def test_eot_id_after_closing_paren_does_not_enter_type_content(self):
        # A real step-10 output (thought translated, action and token verbatim)
        raw = ("Thought: The list has scrolled to the letters G and H. I need to keep scrolling to P to find Peru. "
               "Let me try typing into the search box to filter. First click inside the search box.\n\n"
               "Action: type(content='Peru')\n<|eot_id|>")
        self.assertEqual(action_of(raw), "type(content='Peru')")

    def test_end_of_turn_variant(self):
        # A real step-7 output: another vendor's end token, same damage
        raw = ("Thought: I have clicked the Filter box for Month. Now I need to type \"2022 June\" to filter for June 2022.\n\n"
               "Action: type(content='2022 June')\n<|end_of_turn|>")
        self.assertEqual(action_of(raw), "type(content='2022 June')")

    def test_unknown_future_token_also_stripped(self):
        """The criterion is a regex, not a fixed list: adding tokens one at a time (as was done for `<|im_end|>`) always left others uncovered.

        This uses a token we have never seen in a run, and it must be stripped all the same.
        """
        raw = "Thought: t\n\nAction: type(content='Tokyo')\n<|im_start|>assistant"
        self.assertEqual(action_of(raw), "type(content='Tokyo')")


class TestDamageChannelTwo(unittest.TestCase):
    """2. The self-reinforcing path: the string that goes into history_responses must be clean."""

    def test_output_carries_no_special_token(self):
        raw = "Thought: t\n\nAction: scroll(start_box='<|box_start|>(9,9)<|box_end|>', direction='down')\n<|eot_id|>"
        out = truncate(raw)
        self.assertNotIn("<|eot_id|>", out)
        # but the coordinate syntax must be intact; see the next test class
        self.assertIn("<|box_start|>", out)


class TestBoxTokensSurvive(unittest.TestCase):
    """The coordinate syntax must not be touched. Stripping these two tokens = every click loses its target."""

    def test_click_coordinates_unchanged_with_and_without_trailing_token(self):
        clean = "Thought: t\n\nAction: click(start_box='<|box_start|>(363, 346)<|box_end|>')"
        dirty = clean + "\n<|eot_id|>"
        self.assertEqual(action_of(clean), "click(start_box='<|box_start|>(363,346)<|box_end|>')")
        self.assertEqual(action_of(dirty), action_of(clean))


class TestNoRegressionOnCleanOutput(unittest.TestCase):
    """Regression guard: for normal output this fix must be the identity.

    An offline replay over **7,154 clean steps** from all past runs found zero changes; a few representative
    shapes are kept here so anyone touching this code later finds out immediately.
    """

    CLEAN = [
        "Thought: click once.\n\nAction: click(start_box='<|box_start|>(1,2)<|box_end|>')",
        "Thought: submit the answer.\n\nAction: finished(content='34')",
        "Thought: look for it.\n\nAction: find_text(text='Peru')",
        "Thought: choose.\n\nAction: select(start_box='<|box_start|>(3,4)<|box_end|>', option='Vietnam')",
    ]

    def test_identity(self):
        for raw in self.CLEAN:
            with self.subTest(raw=raw[:40]):
                self.assertEqual(truncate(raw), raw.strip())

    def test_existing_fabricated_trajectory_guard_still_works(self):
        """Its original job (cutting off turns the model invented) must still work after this change."""
        raw = ("Thought: click first.\n\nAction: click(start_box='<|box_start|>(1,2)<|box_end|>')\n"
               "<environment>The screenshot has changed after the action.</environment>\n"
               "Thought: now I know the answer.\n\nAction: finished(content='14')")
        self.assertNotIn("finished", truncate(raw))


if __name__ == "__main__":
    unittest.main()
