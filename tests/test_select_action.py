"""Native <select> action: the whole chain from model output to the selected element (no model calls).

Why a dedicated action: a native `<select>` opens an **OS-level popup** that is not in the page's
render tree. It never shows in the screenshot and cannot be hit with page coordinates. One task died
exactly here: the same spot clicked 9 times, then a double click and keyboard input, all without effect,
until all 40 steps were gone.

The chain has three parts, and only the third needs a real browser:

    model output string --parse_action--> action dict --select_option_at--> element selected
    \____________ covered here (pure functions + stand-ins, zero tokens) ___________/

The third part (whether Playwright can select a real site's <select>) was verified mechanically on a
real page. The only thing left that cannot be verified offline is "will the model emit select() when
it should", which is a question for the prompt, not for this code.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

from fakes import FakeElement, FakePage  # noqa: E402
from web_controller import parse_action, select_option_at  # noqa: E402


def opt(text, value=None):
    return {"text": text, "value": value if value is not None else text.lower()}


class TestParsing(unittest.TestCase):
    """Part one: the model's action string -> action dict."""

    def test_parses_box_format_emitted_by_the_model(self):
        action = parse_action(
            "select(start_box='<|box_start|>(512,344)<|box_end|>', option='Viet Nam')")
        self.assertEqual(action["name"], "Select")
        self.assertEqual(action["coordinate"], [512, 344])
        self.assertEqual(action["value"], "Viet Nam")

    def test_parses_double_quoted_option(self):
        action = parse_action(
            'select(start_box=\'<|box_start|>(10,20)<|box_end|>\', option="Germany")')
        self.assertEqual(action["value"], "Germany")

    def test_missing_option_is_not_silently_treated_as_a_click(self):
        """Missing arguments must end up as Unknown. If it degraded into another action, the model would get the false
        feedback "done, but no effect"; a silently failing action is far more dangerous than one that reports an error
        (the hotkey idling is the cautionary tale)."""
        action = parse_action("select(start_box='<|box_start|>(10,20)<|box_end|>')")
        self.assertEqual(action["name"], "Unknown")


class TestOptionMatching(unittest.TestCase):
    """Part two: coordinates -> find the <select> -> choose the option."""

    def make_page(self, options, element=True):
        return FakePage("p", element=FakeElement() if element else None, options=options)

    def test_exact_text_match(self):
        page = self.make_page([opt("Germany"), opt("Viet Nam"), opt("France")])
        ok, note = select_option_at(page, 5, 5, "Viet Nam")
        self.assertTrue(ok, note)
        self.assertEqual(page.element.selected_index, 1)

    def test_normalized_match_bridges_the_spelling_gap(self):
        """The task and the model write "Vietnam", the site's option says "Viet Nam". The model cannot see the popup
        and has no way to learn the right spelling; without normalization it just retries until the steps run out."""
        page = self.make_page([opt("Germany"), opt("Viet Nam")])
        for spelling in ("Vietnam", "vietnam", "VIET-NAM", "  viet nam  "):
            page.element = FakeElement()
            ok, note = select_option_at(page, 5, 5, spelling)
            self.assertTrue(ok, f"{spelling!r}: {note}")
            self.assertEqual(page.element.selected_index, 1)

    def test_matches_on_value_not_just_visible_text(self):
        page = self.make_page([opt("United States", "us"), opt("Germany", "de")])
        ok, _ = select_option_at(page, 5, 5, "de")
        self.assertTrue(ok)
        self.assertEqual(page.element.selected_index, 1)

    def test_no_native_select_at_point_tells_the_model_to_click_instead(self):
        """On a custom div dropdown the right advice is "just click it", not "keep trying select()".
        The more specific the feedback, the less the model idles next."""
        page = self.make_page([], element=False)
        ok, note = select_option_at(page, 5, 5, "Anything")
        self.assertFalse(ok)
        self.assertIn("click", note.lower())

    def test_unmatched_option_returns_the_option_list(self):
        """On failure, return the list of options: the model cannot see the popup, guessing option names is
        all it can do, and handing it the list is far cheaper than letting it guess."""
        page = self.make_page([opt("Germany"), opt("France"), opt("Viet Nam")])
        ok, note = select_option_at(page, 5, 5, "Atlantis")
        self.assertFalse(ok)
        self.assertIsNone(page.element.selected_index)
        for expected in ("Germany", "France", "Viet Nam"):
            self.assertIn(expected, note)

    def test_long_option_list_is_truncated_with_a_count(self):
        """Country / year dropdowns often have hundreds of entries; returning all of them would crowd out the observation."""
        page = self.make_page([opt(f"Country {i}") for i in range(60)])
        ok, note = select_option_at(page, 5, 5, "Atlantis")
        self.assertFalse(ok)
        self.assertIn("+35 more", note)
        self.assertNotIn("Country 40", note)

    def test_empty_option_list_reads_as_still_loading(self):
        """The <select> exists but its options are filled in asynchronously. A real case that an end-to-end
        check once missed: the site's JS had not finished loading, and select never fired."""
        page = self.make_page([])
        ok, note = select_option_at(page, 5, 5, "Viet Nam")
        self.assertFalse(ok)
        self.assertIn("loading", note.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
