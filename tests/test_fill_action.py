"""fill(): fill text into the input box at the coordinates. Offline, zero tokens.

It targets **unreliable input**, one cause of tasks running out of steps: in one development
set **15 of 70 tasks typed exactly the same content more than once** (= the previous input did
not take); one task failed once on each of its four fields, the model saying "the box already
has focus (blue border), but the click produced no detectable change".

The root cause is the old type(): `excute_action` hard-codes `cx, cy = -1, -1`, so ctrl+a / Backspace /
keystrokes all go to **whatever has focus**, which depends entirely on the previous click,
and a slightly missed click **is invisible in the screenshot**.

The two things this file must guard above all:
  1. **The fallback path stays open.** The biggest risk of a new action is not that it works
     badly but that it leads the model away from a path that was working. When fill() finds no
     input box it must say plainly "click it, then use type()", not just report a failure.
     TestFallbackStaysOpen guards this.
  2. **Action space / parse_action_type / parse_action / excute_action line up.**
     When back() was added only to parse_action_type, all 6 of the model's calls fell into
     Unknown and the executor never ran: a silent failure, very hard to trace. TestFourWayAlignment guards this.

⚠️ Coverage boundary: the stand-ins here verify the **Python side** (frame offset conversion,
   deeper frames win, clear-then-type, both newline forms, failure feedback). The
   `elementFromPoint` + `closest/querySelector` **JS can only be verified in a real browser**;
   that was done separately on fixed local file:// pages.
"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

sys.path.insert(0, os.path.dirname(__file__))

import prompts  # noqa: E402
import runtime_config  # noqa: E402
import web_controller as W  # noqa: E402
from fakes import FakeContext, FakePage as BaseFakePage  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402


# ── Stand-ins ──────────────────────────────────────────────────────────────
# Only the methods fill_at actually uses. The DOM is reduced to a small "rectangle -> element"
# model, because what is under test is coordinate conversion and priority, not hit testing.

class FakeInput:
    """Input stand-in: records the effect of fill/keyboard in value, to assert "replaced, not appended"."""

    def __init__(self, value="", kind="input", fill_raises=False, click_raises=False):
        self.value = value
        self.kind = kind
        self.fill_raises = fill_raises
        self.click_raises = click_raises
        self.clicked = self.focused = self.scrolled = False

    def scroll_into_view_if_needed(self, timeout=None):
        self.scrolled = True

    def click(self, timeout=None):
        if self.click_raises:
            raise RuntimeError("element is covered by another element")
        self.clicked = True

    def focus(self):
        self.focused = True

    def fill(self, text):
        if self.fill_raises:
            raise RuntimeError("not an <input>")
        self.value = text


class FakeKeyboard:
    def __init__(self, page):
        self.page = page
        self.pressed = []

    def type(self, text, delay=None):
        el = self.page.focused_element
        if el is not None:
            el.value += text

    def press(self, key):
        self.pressed.append(key)
        el = self.page.focused_element
        if el is None:
            return
        if key == "Control+a":
            el._selected_all = True
        elif key == "Backspace" and getattr(el, "_selected_all", False):
            el.value = ""


class FakeHandle:
    def __init__(self, el):
        self._el = el

    def as_element(self):
        return self._el


class FakeFrame:
    def __init__(self, page, parent=None, box=None, hits=None):
        self.page = page
        self._parent = parent
        self._box = box                      # relative to the main frame's viewport, as in Playwright
        self.hits = hits or []               # [((x0,y0,x1,y1), element_or_None)]
        self.seen = []                       # local coordinates this frame received

    @property
    def parent_frame(self):
        return self._parent

    def frame_element(self):
        outer = self

        class _FE:
            def bounding_box(self_inner):
                return outer._box
        return _FE()

    def evaluate_handle(self, js, arg):
        x, y, _sel = arg
        self.seen.append((x, y))
        for (x0, y0, x1, y1), el in self.hits:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return FakeHandle(el)
        return FakeHandle(None)


class FakePage:
    def __init__(self, frames=None, readback=None):
        self.frames = frames if frames is not None else []
        self.keyboard = FakeKeyboard(self)
        self.focused_element = None
        self.waited_ms = 0
        self._readback = readback            # None = use the element's own value

    def wait_for_timeout(self, ms):
        self.waited_ms += ms

    def evaluate(self, js, el=None):
        if self._readback is not None:
            return self._readback
        return el.value if el is not None else None


class FillablePage(BaseFakePage):
    """Stand-in for excute_action: the shared FakePage plus frames / keyboard."""

    def __init__(self, *a, element=None, **kw):
        kw.setdefault("context", FakeContext())
        super().__init__(*a, **kw)
        self.keyboard = FakeKeyboard(self)
        self.focused_element = element
        self.frames = [FakeFrame(self, parent=None, hits=[((0, 0, 1920, 1080), element)])]

    def evaluate(self, script, arg=None):
        if hasattr(arg, "value"):
            return arg.value
        return super().evaluate(script, arg)


def _page_with(el, box=(0, 0, 1920, 1080), readback=None):
    page = FakePage(readback=readback)
    main = FakeFrame(page, parent=None, hits=[(box, el)])
    page.frames = [main]
    return page


def _fill(page, x, y, content):
    """After click/focus in fill_at, focus should be on that element; wire that up by hand in the stand-in."""
    orig = W.editable_at

    def spy(p, xx, yy):
        el = orig(p, xx, yy)
        p.focused_element = el
        return el
    W.editable_at = spy
    try:
        return W.fill_at(page, x, y, content, timeout_ms=100)
    finally:
        W.editable_at = orig


# ── Tests ──────────────────────────────────────────────────────────────────

class TestFourWayAlignment(unittest.TestCase):
    """Action space / parse_action_type / parse_action / excute_action must all know it."""

    ACTION = "fill(start_box='<|box_start|>(484,859)<|box_end|>', content='20')"

    def test_action_space_contains_fill_only_when_enabled(self):
        on = prompts.normal_action_space(True, True, True, enable_fill=True)
        off = prompts.normal_action_space(True, True, True, enable_fill=False)
        self.assertIn("fill(start_box=", on)
        self.assertNotIn("fill(start_box=", off)

    def test_fill_line_precedes_type_line(self):
        """Order carries meaning: "fill when there is a box, type only when there is none" must read right in the action space itself."""
        sp = prompts.normal_action_space(True, True, True, enable_fill=True)
        self.assertLess(sp.index("fill(start_box="), sp.index("type(content=''"))

    def test_parse_action_type(self):
        self.assertEqual(W.parse_action_type(self.ACTION), "Fill")

    def test_parse_action(self):
        self.assertEqual(W.parse_action(self.ACTION),
                         {"name": "Fill", "coordinate": [484, 859], "value": "20"})

    def test_parse_action_does_not_shadow_type(self):
        """The fill branch must not swallow type(): both contain content=."""
        self.assertEqual(W.parse_action("type(content='20')"),
                         {"name": "Type", "value": "20"})


class TestLocating(unittest.TestCase):
    def test_finds_element_at_point(self):
        el = FakeInput()
        page = _page_with(el)
        self.assertIs(W.editable_at(page, 100, 200), el)

    def test_iframe_offset_is_subtracted(self):
        """An input inside an iframe: coordinates must be converted to the frame's local coordinates.

        In one task the model said "the search box is an iframe and cannot be clicked directly"
        and ran out of all 40 steps; without iframe handling such tasks are lost.
        """
        el = FakeInput()
        page = FakePage()
        main = FakeFrame(page, parent=None, hits=[((0, 0, 1920, 1080), None)])
        inner = FakeFrame(page, parent=main, box={"x": 300, "y": 200, "width": 400, "height": 300},
                          hits=[((50, 50, 120, 80), el)])
        page.frames = [main, inner]
        # page coordinates (360, 260) -> frame-local (60, 60)
        self.assertIs(W.editable_at(page, 360, 260), el)
        self.assertIn((60, 60), inner.seen)

    def test_point_outside_iframe_does_not_enter_it(self):
        el = FakeInput()
        page = FakePage()
        main = FakeFrame(page, parent=None, hits=[((0, 0, 1920, 1080), None)])
        inner = FakeFrame(page, parent=main, box={"x": 300, "y": 200, "width": 400, "height": 300},
                          hits=[((0, 0, 9999, 9999), el)])
        page.frames = [main, inner]
        self.assertIsNone(W.editable_at(page, 50, 50))
        self.assertEqual(inner.seen, [])

    def test_deepest_frame_wins(self):
        outer_el, inner_el = FakeInput("outer"), FakeInput("inner")
        page = FakePage()
        main = FakeFrame(page, parent=None, hits=[((0, 0, 1920, 1080), outer_el)])
        inner = FakeFrame(page, parent=main, box={"x": 0, "y": 0, "width": 800, "height": 600},
                          hits=[((0, 0, 800, 600), inner_el)])
        page.frames = [main, inner]
        self.assertIs(W.editable_at(page, 100, 100), inner_el)

    def test_broken_frame_is_skipped_not_fatal(self):
        """Cross-origin or navigating frames raise. That must not bring down the whole action."""
        el = FakeInput()
        page = FakePage()
        main = FakeFrame(page, parent=None, hits=[((0, 0, 1920, 1080), el)])

        class Boom(FakeFrame):
            def frame_element(self):
                raise RuntimeError("cross-origin")
        bad = Boom(page, parent=main, box=None)
        page.frames = [main, bad]
        self.assertIs(W.editable_at(page, 10, 10), el)


class TestReplaceNotAppend(unittest.TestCase):
    """Guards the four-field failure: the old ctrl+a hit the wrong focus and the new value was **appended** to the old one."""

    def test_existing_value_is_cleared_first(self):
        el = FakeInput(value="6/30/2026")
        page = _page_with(el)
        ok, note = _fill(page, 10, 10, "10/31/2019")
        self.assertTrue(ok)
        self.assertEqual(el.value, "10/31/2019")

    def test_falls_back_to_keyboard_clear_when_fill_unsupported(self):
        """Custom components may reject el.fill(). Fall back to select-all + delete; focus is already right by then."""
        el = FakeInput(value="old", fill_raises=True)
        page = _page_with(el)
        ok, _ = _fill(page, 10, 10, "new")
        self.assertTrue(ok)
        self.assertEqual(el.value, "new")
        self.assertIn("Control+a", page.keyboard.pressed)

    def test_focus_fallback_when_click_is_covered(self):
        el = FakeInput(click_raises=True)
        page = _page_with(el)
        ok, _ = _fill(page, 10, 10, "x")
        self.assertTrue(ok)
        self.assertTrue(el.focused)


class TestSubmitNewline(unittest.TestCase):
    """Both forms of a trailing newline must be recognized.

    The action space says `"\\n"`, and the model copies that as **two literal characters**: 79 of
    356 type() calls (22%) had this shape. The old type() path typed it into the box as text
    (`type(content='permit search\\n')`) and never pressed Enter.
    """

    def test_literal_backslash_n_presses_enter(self):
        el = FakeInput()
        page = _page_with(el)
        _fill(page, 10, 10, "Peru\\n")
        self.assertEqual(el.value, "Peru")
        self.assertIn("Enter", page.keyboard.pressed)

    def test_real_newline_presses_enter(self):
        el = FakeInput()
        page = _page_with(el)
        _fill(page, 10, 10, "Peru\n")
        self.assertEqual(el.value, "Peru")
        self.assertIn("Enter", page.keyboard.pressed)

    def test_no_newline_does_not_press_enter(self):
        el = FakeInput()
        page = _page_with(el)
        _fill(page, 10, 10, "Peru")
        self.assertEqual(el.value, "Peru")
        self.assertNotIn("Enter", page.keyboard.pressed)


class TestNativeDateInput(unittest.TestCase):
    """A native <input type=date> accepts only ISO: typing "10/31/2019" key by key **leaves the box empty**
    (confirmed in a real browser), while tasks and the model write M/D/YYYY,
    e.g. an "End Date" field or a date range in a request search.

    The end-to-end half was verified in a real browser; this only pins the conversion table.
    """

    def test_us_format(self):
        self.assertEqual(W._to_iso_date("10/31/2019"), "2019-10-31")
        self.assertEqual(W._to_iso_date("1/5/2019"), "2019-01-05")

    def test_iso_and_slash_iso_pass_through(self):
        self.assertEqual(W._to_iso_date("2019-10-31"), "2019-10-31")
        self.assertEqual(W._to_iso_date("2019/10/31"), "2019-10-31")

    def test_unparseable_returns_none_so_the_model_gets_told(self):
        """When it cannot be recognized, **do not guess**: better to send the model back to rewrite it than to fill in a wrong date."""
        for bad in ("October 2019", "31.10.2019", "", "last Tuesday"):
            self.assertIsNone(W._to_iso_date(bad), bad)


class TestFeedbackToModel(unittest.TestCase):
    """The read-back is information the model has no other way to get; silent failure is exactly how the four-field task died."""

    def test_reports_actual_value_on_success(self):
        el = FakeInput()
        page = _page_with(el)
        ok, note = _fill(page, 10, 10, "Peru")
        self.assertTrue(ok)
        self.assertIn("Peru", note)

    def test_reformatted_value_is_reported_not_treated_as_failure(self):
        """A date box reorders 10/31/2019 into 2019-10-31. That is not a failure, but the model needs to know."""
        el = FakeInput()
        page = _page_with(el, readback="2019-10-31")
        ok, note = _fill(page, 10, 10, "10/31/2019")
        self.assertTrue(ok)
        self.assertIn("2019-10-31", note)


class TestFallbackStaysOpen(unittest.TestCase):
    """The most important category: a new action must not lead the model away from a path that was working.

    This is the regression we worry about: the model sees fill() and stops using click()+type(),
    while fill() cannot reach some controls. Both paths must stay, and a failure must **point back explicitly**.
    """

    def test_no_editable_element_points_back_to_click_and_type(self):
        page = _page_with(None)
        ok, note = _fill(page, 10, 10, "x")
        self.assertFalse(ok)
        self.assertIn("type()", note)
        self.assertIn("click", note.lower())

    def test_disabled_flag_rejects_and_points_back_to_type(self):
        """With the switch off, the executor must reject it and tell the model to use click + type() instead.

        (fill never appears in the action space then, but the model occasionally emits actions it was
        never given, which is exactly how the ineffective hotkeys idled whole runs away. The executor must catch it.)
        """
        page = FillablePage("p", element=None)
        W.LAST_ACTION_NOTE = ""
        with unittest.mock.patch.object(runtime_config, "_CACHED",
                                        RuntimeConfig(enable_fill=False)):
            W.excute_action(page, None,
                            {"name": "Fill", "coordinate": [10, 10], "value": "x"})
        self.assertIn("REJECTED", W.LAST_ACTION_NOTE)
        self.assertIn("type(content=", W.LAST_ACTION_NOTE)

    def test_enabled_flag_reaches_the_element(self):
        el = FakeInput()
        page = FillablePage("p", element=el)
        W.LAST_ACTION_NOTE = ""
        with unittest.mock.patch.object(runtime_config, "_CACHED",
                                        RuntimeConfig(enable_fill=True)):
            W.excute_action(page, None,
                            {"name": "Fill", "coordinate": [10, 10], "value": "Peru"})
        self.assertEqual(el.value, "Peru")
        self.assertIn("Peru", W.LAST_ACTION_NOTE)


if __name__ == "__main__":
    unittest.main()
