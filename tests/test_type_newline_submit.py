"""A trailing newline in `type(content='xxx\\n')` should press Enter, not type `\\` and `n` into the box. Offline, zero tokens.

The action space says: `type(content='') #If you want to submit your input, use "\\n" at the end`.
The model complies and writes **two literal characters**, while `click_type` types character by
character, so the search box really ends up with `\\n` in it and **Enter is never pressed**.

Measured over all past runs: **79 of 356 type() calls (22%)** had this shape, e.g.
`type(content='permit search\\n')`.
Same family as `<|eot_id|>`: **the prompt promised behaviour the executor never implemented.**

But it differs from `<|eot_id|>` in one essential way, and the tests are structured around it:
   `<|eot_id|>` only appears in tasks that fail anyway, so stripping it is the identity on clean output;
   this path, however, **is also taken by tasks that pass** (one passing task used a `\\n` type).
   So it is a **behaviour change** and needs its own switch, so it can be turned off on its own.
   TestFlagOff pins "switched off, the behaviour is exactly as before the fix".
"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

import runtime_config  # noqa: E402
import web_controller as W  # noqa: E402
from fakes import FakeContext, FakePage  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402


class TypeablePage(FakePage):
    """Records what was actually typed and which keys were pressed."""

    def __init__(self, *a, **kw):
        kw.setdefault("context", FakeContext())
        super().__init__(*a, **kw)
        outer = self
        self.typed = []
        self.pressed = []
        self._fp = 0

        class _KB:
            def type(self_kb, text, delay=None):
                outer.typed.append(text)

            def press(self_kb, key):
                outer.pressed.append(key)
        self.keyboard = _KB()

    def evaluate(self, script, arg=None):
        # page_fingerprint expects a dict that differs before and after the action: typing changes the page.
        # The base class returns the string "fp-0", which would make the closing no-op check raise.
        if "elementFromPoint" not in str(script):
            self._fp += 1
            return {"n": self._fp}
        return super().evaluate(script, arg)

    @property
    def text(self):
        """The typed characters joined together: what the box will finally contain."""
        return "".join(self.typed)


def run_type(content, enabled):
    page = TypeablePage("p")
    W.LAST_ACTION_NOTE = ""
    with unittest.mock.patch.object(runtime_config, "_CACHED",
                                    RuntimeConfig(enable_type_newline_submit=enabled)):
        W.excute_action(page, None, {"name": "Type", "value": content})
    return page


class TestSplitter(unittest.TestCase):
    def test_literal_backslash_n(self):
        self.assertEqual(W.split_submit_newline("permit search\\n"), (True, "permit search"))

    def test_real_newline(self):
        self.assertEqual(W.split_submit_newline("permit search\n"), (True, "permit search"))

    def test_no_trailing_newline(self):
        self.assertEqual(W.split_submit_newline("permit search"), (False, "permit search"))

    def test_newline_only_in_the_middle_is_left_alone(self):
        """Only the **trailing** newline. A newline in the middle is part of the text and must not be touched."""
        self.assertEqual(W.split_submit_newline("a\\nb"), (False, "a\\nb"))


class TestFlagOff(unittest.TestCase):
    """Switched off, it must be byte-identical to the behaviour before the fix."""

    def test_literal_backslash_n_is_still_typed_as_text(self):
        page = run_type("permit search\\n", enabled=False)
        self.assertEqual(page.text, "permit search\\n")
        self.assertNotIn("Enter", page.pressed)

    def test_plain_text_unchanged(self):
        page = run_type("Peru", enabled=False)
        self.assertEqual(page.text, "Peru")
        self.assertNotIn("Enter", page.pressed)


class TestFlagOn(unittest.TestCase):
    def test_literal_backslash_n_presses_enter_and_is_not_typed(self):
        page = run_type("permit search\\n", enabled=True)
        self.assertEqual(page.text, "permit search")
        self.assertIn("Enter", page.pressed)

    def test_real_newline_too(self):
        page = run_type("permit search\n", enabled=True)
        self.assertEqual(page.text, "permit search")
        self.assertIn("Enter", page.pressed)

    def test_without_trailing_newline_nothing_changes(self):
        page = run_type("Peru", enabled=True)
        self.assertEqual(page.text, "Peru")
        self.assertNotIn("Enter", page.pressed)

    def test_does_not_occupy_the_stall_feedback_channel(self):
        """A successful Enter must **not** write LAST_ACTION_NOTE.

        The closing no-op check only reports "the page did not change" when there is no note. Taking that
        channel would deprive the model of the "nothing happened" signal, the same trap as the ineffective
        hotkeys that once idled whole runs away.
        """
        page = run_type("permit search\\n", enabled=True)
        self.assertIn("Enter", page.pressed)
        self.assertNotIn("Enter was pressed", W.LAST_ACTION_NOTE)


if __name__ == "__main__":
    unittest.main()
