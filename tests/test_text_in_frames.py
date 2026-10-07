"""find_text() / read_text() descend into iframes. Offline, zero tokens.

**Root cause**: across several runs find_text hit 10 times and missed 18, and most misses were
inside iframes: chart labels in an embedded chart, section titles in a filings viewer, and a country
list that lives entirely in an iframe (one task scrolled for 40 steps looking for "Peru").
`page.get_by_text` only looks at the main document and reports "not found" for all of them, leaving
the model to scroll screen by screen, which is exactly what find_text exists to replace.

Two things to guard:
  1. **Main frame first**: when the main frame has a match, no child frame is touched and behaviour is exactly as before.
  2. The boundary stays: the find_text branch still only locates and scrolls (the source scan in test_find_text_action keeps guarding it).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

import web_controller as wc  # noqa: E402
from fakes import FakeContext, FakePage  # noqa: E402


class FakeLocator:
    def __init__(self, n, text="Asia Pacific"):
        self._n = n
        self._text = text
        self.scrolled = False
        self.scrolled_js = False

    def count(self):
        return self._n

    @property
    def first(self):
        return self

    def nth(self, i):
        self.picked = i
        return self

    def scroll_into_view_if_needed(self, timeout=None):
        self.scrolled = True

    def evaluate(self, js, arg=None):
        if "scrollIntoView" in js:
            self.scrolled_js = True
        return None

    def inner_text(self):
        return self._text

    def locator(self, sel):
        return FakeLocator(1, text=self._text + " (parent row with more context text here ok)")


class TextFrame:
    def __init__(self, parent, texts, raise_on_query=False):
        self._parent = parent
        self.texts = texts
        self.raise_on_query = raise_on_query
        self.queried = []
        self.last_locator = None

    @property
    def parent_frame(self):
        return self._parent

    def get_by_text(self, text, exact=False):
        self.queried.append(text)
        if self.raise_on_query:
            raise RuntimeError("Frame was detached")
        hits = [t for t in self.texts if text.lower() in t.lower()]
        self.last_locator = FakeLocator(len(hits), text=hits[0] if hits else "")
        return self.last_locator


class TextPage(FakePage):
    def __init__(self, main_texts, child_texts=(), child_raises=False):
        super().__init__(context=FakeContext())
        self.main = TextFrame(None, main_texts)
        self.child = TextFrame(self.main, child_texts, raise_on_query=child_raises)
        self.frames = [self.main, self.child]

    def get_by_text(self, text, exact=False):
        return self.main.get_by_text(text, exact)


class TestLocateText(unittest.TestCase):
    def test_main_frame_hit_never_touches_child_frames(self):
        page = TextPage(main_texts=["Asia Pacific"], child_texts=["Asia Pacific"])
        loc, in_frame, n = wc.locate_text(page, "Asia Pacific")
        self.assertEqual((in_frame, n), (False, 1))
        self.assertEqual(page.child.queried, [], "a match in the main frame must not touch child frames")

    def test_falls_through_to_child_frame(self):
        page = TextPage(main_texts=["Global M&A"], child_texts=["Global", "Asia Pacific", "EMEA"])
        loc, in_frame, n = wc.locate_text(page, "asia pacific")
        self.assertEqual((in_frame, n), (True, 1))

    def test_nothing_anywhere(self):
        page = TextPage(main_texts=["x"], child_texts=["y"])
        self.assertEqual(wc.locate_text(page, "Peru"), (None, False, 0))

    def test_detached_frame_is_skipped(self):
        page = TextPage(main_texts=["x"], child_texts=["Peru"], child_raises=True)
        self.assertEqual(wc.locate_text(page, "Peru"), (None, False, 0))

    def test_pages_without_frames_still_work(self):
        page = TextPage(main_texts=["x"])
        page.frames = None
        self.assertEqual(wc.locate_text(page, "Peru"), (None, False, 0))


class TestActions(unittest.TestCase):
    def setUp(self):
        wc.reset_stall_counters()
        wc.LAST_ACTION_NOTE = ""

    def test_find_text_scrolls_to_text_inside_iframe_and_says_so(self):
        page = TextPage(main_texts=["report"], child_texts=["Asia Pacific"])
        wc.excute_action(page, None, {"name": "FindText", "value": "Asia Pacific"})
        note = wc.take_action_note()
        self.assertIn("scrolled to 'Asia Pacific'", note)
        self.assertIn("embedded frame", note)
        self.assertEqual(wc.find_text_count(), 1, "a match found inside an iframe is still a real scroll and counts against the budget")
        loc = page.child.last_locator
        self.assertTrue(loc.scrolled_js, "elements in cross-origin iframes must use JS scrollIntoView (Playwright's times out)")
        self.assertFalse(loc.scrolled)

    def test_find_text_on_main_frame_keeps_old_wording(self):
        page = TextPage(main_texts=["Asia Pacific"], child_texts=["Asia Pacific"])
        wc.excute_action(page, None, {"name": "FindText", "value": "Asia Pacific"})
        note = wc.take_action_note()
        self.assertIn("scrolled to 'Asia Pacific'", note)
        self.assertNotIn("embedded frame", note)
        self.assertTrue(page.main.last_locator.scrolled, "a main-frame match still uses Playwright's own method")

    def test_find_text_miss_everywhere_is_not_charged(self):
        page = TextPage(main_texts=["a"], child_texts=["b"])
        wc.excute_action(page, None, {"name": "FindText", "value": "Peru"})
        self.assertIn("found no text", wc.take_action_note())
        self.assertEqual(wc.find_text_count(), 0)

    def test_read_text_reads_from_iframe_and_says_so(self):
        page = TextPage(main_texts=["report"], child_texts=["Health services: 502"])
        wc.excute_action(page, None, {"name": "ReadText", "value": "Health services"})
        note = wc.take_action_note()
        self.assertIn("Health services: 502", note)
        self.assertIn("embedded frame", note)


class TestNextOccurrence(unittest.TestCase):
    """Searching the same text again = jump to the next match. In long documents the first match is often a
    passing reference while the real table is the 3rd or 4th; the model was taken back to the first match three times."""

    def setUp(self):
        wc.reset_stall_counters()
        wc.LAST_ACTION_NOTE = ""

    def _find(self, page, text):
        wc.excute_action(page, None, {"name": "FindText", "value": text})
        return wc.take_action_note()

    def test_repeating_the_same_text_advances_and_wraps(self):
        page = TextPage(main_texts=["Commitments and Contingencies"] * 4)
        n1 = self._find(page, "Commitments and Contingencies")
        self.assertEqual(page.main.last_locator.picked, 0)
        self.assertIn("occurrence 1 of 4", n1)
        self.assertIn("same text again", n1)
        self._find(page, "Commitments and Contingencies")
        self.assertEqual(page.main.last_locator.picked, 1)
        self._find(page, "Commitments and Contingencies")
        self._find(page, "Commitments and Contingencies")
        self.assertEqual(page.main.last_locator.picked, 3)
        self._find(page, "Commitments and Contingencies")
        self.assertEqual(page.main.last_locator.picked, 0, "wrap around to the first match at the end")

    def test_different_text_starts_from_the_first_occurrence(self):
        page = TextPage(main_texts=["Note 16", "Note 16", "Thereafter", "Thereafter"])
        self._find(page, "Note 16"); self._find(page, "Note 16")
        self.assertEqual(page.main.last_locator.picked, 1)
        self._find(page, "Thereafter")
        self.assertEqual(page.main.last_locator.picked, 0)

    def test_single_hit_note_has_no_next_hint(self):
        page = TextPage(main_texts=["Asia Pacific"])
        self.assertNotIn("occurrence", self._find(page, "Asia Pacific"))

    def test_counter_clears_between_tasks(self):
        page = TextPage(main_texts=["x", "x"])
        self._find(page, "x"); self._find(page, "x")
        wc.reset_stall_counters()
        self._find(page, "x")
        self.assertEqual(page.main.last_locator.picked, 0)


if __name__ == "__main__":
    unittest.main()
