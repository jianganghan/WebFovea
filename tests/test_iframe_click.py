"""A click landing on an iframe: child-frame fingerprint + DOM click fallback. Offline, zero tokens.

**Root cause** (measured with an embedded chart):
  1. The page fingerprint used for no-op detection only covers the main frame. A tab switch
     inside the iframe succeeded (y-axis 6,000 -> 2,000) while the main frame's url / title /
     node count / text length did not move, so the model was told "the click had no effect, this is
     ineffective action #N, change strategy": **false feedback** that talked it off the right path.
  2. With the tab scrolled to the bottom of the viewport (y ~ 917 of 1080), three physical CDP
     clicks in a row were swallowed, while one DOM event inside the iframe worked. In one task the
     agent found the right report by step 50 and spent the remaining 30 steps on this.

The two things this file must guard above all:
  1. **Clicks on the main frame are not changed at all**: for clicks outside iframes, no-op
     detection behaves exactly as before (TestMainFrameUntouched).
  2. **The fallback only acts when neither the main frame nor the child frame changed**: if the
     physical click worked there is no second click, since clicking twice is a disaster for toggles
     (TestIframeClick.test_effective_physical_click_is_left_alone).

⚠️ Coverage boundary: this verifies the Python-side decisions and the fallback order. The JS in
   `_JS_CLICK_AT` and the "physical clicks swallowed at the bottom of the viewport" effect itself can
   only be verified in a real browser; that was done on a local file:// page embedding the chart.
"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

import web_controller as wc  # noqa: E402
from fakes import FakeContext, FakePage  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402


# ── Stand-ins ──────────────────────────────────────────────────────────────

def _fp(marker, thash=0):
    """A valid fingerprint: only nodes varies with marker, everything else is fixed; thash is the body-text hash."""
    return {"url": "u", "title": "t", "nodes": marker, "text": 1, "sx": 0, "sy": 0,
            "inner": 0, "active": "", "thash": thash}


class FakeFrame:
    """Playwright Frame stand-in: a rectangle (relative to the main viewport) plus its own state counter.

    `state` is the frame's "content version": an effective physical or DOM click adds 1 and the
    fingerprint follows. That is all the code under test has to decide whether an action had an effect.
    """

    def __init__(self, parent=None, box=None, physical_click_works=False,
                 dom_click_works=False, tag="SPAN", hover_changes_text=False,
                 url="https://flo.uri.sh/visualisation/1/embed"):
        self._parent = parent
        self._box = box
        self.url = url
        self.state = 0
        self.thash = 0                   # body-text hash: a same-length tooltip swap changes only this
        self.tag = tag
        self.physical_click_works = physical_click_works
        self.dom_click_works = dom_click_works
        self.hover_changes_text = hover_changes_text
        self.dom_clicks = []             # DOM clicks received (frame-local coordinates)
        self.dom_hovers = []             # DOM moves received (frame-local coordinates)
        self.dom_hover_works = True

    @property
    def parent_frame(self):
        return self._parent

    def frame_element(self):
        box = self._box

        class _FE:
            def bounding_box(self_inner):
                return box
        return _FE()

    def contains(self, x, y):
        b = self._box
        return b["x"] <= x <= b["x"] + b["width"] and b["y"] <= y <= b["y"] + b["height"]

    def evaluate(self, js, arg=None):
        if "dispatchEvent" in js and "'click'" in js:        # _JS_CLICK_AT
            self.dom_clicks.append(tuple(arg))
            if self.dom_click_works:
                self.state += 1
            return {"tag": "SPAN", "text": "Asia Pacific"}
        if "dispatchEvent" in js and "mousemove" in js:      # _JS_HOVER_AT
            self.dom_hovers.append(tuple(arg))
            if self.dom_hover_works:
                self.thash += 1                              # tooltip appears = body text changed
            return {"tag": "path", "text": ""}
        if "elementFromPoint" in js:                 # _DESCRIBE_POINT_JS
            return {"tag": self.tag, "cls": "tab", "text": "Asia Pacific",
                    "clickable": False, "disabled": False}
        return _fp(self.state, self.thash)           # _FP_JS


class FakeMouse:
    def __init__(self, page):
        self.page = page
        self.clicks = []

    def click(self, x, y, **kw):
        self.clicks.append((x, y))
        for f in (self.page.frames or []):
            if f.parent_frame is not None and f.contains(x, y):
                # In a real browser, clicking into an iframe always moves the main frame's focus onto the
                # <iframe>, so the main fingerprint's active changes, even if the click itself was swallowed.
                # The stand-in must reproduce that, or the "focus move counted as a reaction" false positive cannot be tested.
                self.page.fingerprint = dict(self.page.fingerprint, active="IFRAME")
                if f.physical_click_works:
                    f.state += 1


class FakeClient:
    """CDP session stand-in: cdp_mouse_move only needs send() not to raise. hover goes through it."""

    def __init__(self, page=None):
        self.page = page

    def send(self, method, *a, **kw):
        if self.page is not None and method == "Input.dispatchMouseEvent":
            x, y = a[0]["x"], a[0]["y"]
            for f in (self.page.frames or []):
                if f.parent_frame is not None and f.contains(x, y) and f.hover_changes_text:
                    f.thash = hash((x, y)) & 0xffff or 1
            if self.page.hover_changes_main_text:
                self.page.fingerprint = dict(self.page.fingerprint, thash=self.page.fingerprint["thash"] + 1)
        return {"result": {"value": {"x": 0, "y": 0}}}


class ClickablePage(FakePage):
    def __init__(self, frames=None):
        super().__init__(context=FakeContext())
        self.frames = frames if frames is not None else []
        self.mouse = FakeMouse(self)
        self.fingerprint = _fp("main")            # main-frame fingerprint: its values never change in this file (except thash)
        self.hover_changes_main_text = False


def _page_with_iframe(**frame_kw):
    page = ClickablePage()
    main = FakeFrame(parent=None)
    child = FakeFrame(parent=main, box={"x": 500, "y": 800, "width": 600, "height": 200}, **frame_kw)
    page.frames = [main, child]
    return page, child


IN_FRAME = (600, 850)        # inside the child frame, local coordinates (100, 50)
OUTSIDE = (100, 100)         # empty space in the main frame


def _click(page, xy):
    wc.excute_action(page, FakeClient(), {"name": "LeftClick", "coordinate": list(xy)})
    return wc.take_action_note()


class _Base(unittest.TestCase):
    def setUp(self):
        wc.reset_stall_counters()
        wc.LAST_ACTION_NOTE = ""


# ── Clicks on the main frame: behaviour unchanged ──────────────────────────

class TestMainFrameUntouched(_Base):
    def test_stall_on_main_frame_is_reported_exactly_as_before(self):
        page, child = _page_with_iframe(physical_click_works=True, dom_click_works=True)
        note = _click(page, OUTSIDE)
        self.assertIn("NO detectable change", note)
        self.assertEqual(wc.no_change_streak(), 1)
        self.assertEqual(child.dom_clicks, [], "a click on the main frame must never trigger a click inside an iframe")
        self.assertNotIn("embedded frame", note)

    def test_pages_without_frames_attribute_still_work(self):
        """Other tests' FakePage has no frames, and enumerating frames may raise in a real browser too.
        This path may only add value, never break the click itself."""
        page = FakePage(context=FakeContext())
        page.mouse = FakeMouse(page)
        page.frames = None            # iterating None raises TypeError, which must be swallowed
        page.fingerprint = _fp("main")
        note = _click(page, OUTSIDE)
        self.assertIn("NO detectable change", note)


# ── Clicks inside an iframe ────────────────────────────────────────────────

class TestIframeClick(_Base):
    def test_change_inside_iframe_is_not_a_stall(self):
        """Root cause 1: main frame unchanged but child frame changed = the action worked; no no-op report."""
        page, child = _page_with_iframe(physical_click_works=True)
        note = _click(page, IN_FRAME)
        self.assertEqual(note, "")
        self.assertEqual(wc.no_change_streak(), 0)

    def test_effective_physical_click_is_left_alone(self):
        """If the physical click worked, no second click: clicking twice would flip a toggle back."""
        page, child = _page_with_iframe(physical_click_works=True, dom_click_works=True)
        _click(page, IN_FRAME)
        self.assertEqual(child.dom_clicks, [])
        self.assertEqual(child.state, 1)

    def test_swallowed_click_falls_back_to_dom_click_in_frame_coordinates(self):
        """Root cause 2: the physical click was swallowed -> one DOM click inside the child frame, in **local coordinates**."""
        page, child = _page_with_iframe(physical_click_works=False, dom_click_works=True)
        note = _click(page, IN_FRAME)
        self.assertEqual(child.dom_clicks, [(100, 50)], "coordinates must have the iframe offset subtracted")
        self.assertEqual(note, "")
        self.assertEqual(wc.no_change_streak(), 0)
        self.assertGreaterEqual(page.waited_ms, 1500, "wait a moment after the fallback click before comparing fingerprints")

    def test_focus_moving_into_the_iframe_is_not_mistaken_for_a_response(self):
        """Confirmed in a real browser: a swallowed click still moves the main frame's focus into the <iframe>,
        changing the main fingerprint's active. Before the fix that counted as "the page reacted", so the
        fallback never ran and the model got no feedback at all."""
        page, child = _page_with_iframe(physical_click_works=False, dom_click_works=False)
        note = _click(page, IN_FRAME)
        self.assertEqual(page.fingerprint["active"], "IFRAME", "the stand-in did not reproduce the focus move; the test is void")
        self.assertEqual(len(child.dom_clicks), 1, "a focus move is not an effect; the fallback must be tried")
        self.assertIn("NO detectable change", note)

    def test_fallback_that_also_fails_reports_stall_with_frame_hint(self):
        page, child = _page_with_iframe(physical_click_works=False, dom_click_works=False)
        note = _click(page, IN_FRAME)
        self.assertEqual(len(child.dom_clicks), 1)
        self.assertIn("NO detectable change", note)
        self.assertIn("embedded frame", note, "the model must be told it clicked something inside an embedded frame")
        self.assertIn("<SPAN", note, "the description must be of the element **inside** the iframe, not the <IFRAME> itself")
        self.assertEqual(wc.no_change_streak(), 1)

    def test_only_left_click_gets_the_dom_fallback(self):
        """A double click has no DOM fallback (a single click would mean something else), but the child-frame fingerprint still counts."""
        page, child = _page_with_iframe(physical_click_works=False, dom_click_works=True)
        wc.excute_action(page, FakeClient(), {"name": "DoubleClick", "coordinate": list(IN_FRAME)})
        self.assertEqual(child.dom_clicks, [])
        self.assertIn("NO detectable change", wc.take_action_note())

    def test_switch_off_restores_old_behaviour(self):
        """Kill switch: with it off, child frames are ignored and nothing is re-clicked, exactly as before the fix,
        when "focus moved into the iframe = the page reacted": a swallowed click was neither reported nor retried."""
        page, child = _page_with_iframe(physical_click_works=False, dom_click_works=True)
        cfg = RuntimeConfig(enable_iframe_click_fallback=False)
        with unittest.mock.patch.object(wc, "get_config", return_value=cfg):
            note = _click(page, IN_FRAME)
        self.assertEqual(child.dom_clicks, [], "with the switch off there must never be a fallback click")
        self.assertEqual(child.state, 0, "the click was swallowed and, with the switch off, stays swallowed")
        self.assertEqual(note, "", "before the fix this was silent (the active change counted as a reaction)")

    def test_captcha_provider_frames_never_get_synthetic_events(self):
        """Turnstile / reCAPTCHA accept only trusted events; synthetic ones do not work and are exactly what bot detection looks for.
        The physical click still happens, but there is no second click and the child-frame fingerprint is not used."""
        page, child = _page_with_iframe(physical_click_works=False, dom_click_works=True,
                                        url="https://challenges.cloudflare.com/cdn-cgi/challenge-platform/turnstile")
        note = _click(page, IN_FRAME)
        self.assertEqual(child.dom_clicks, [], "never dispatch synthetic clicks inside a CAPTCHA frame")
        self.assertEqual(len(page.mouse.clicks), 1, "the physical click still happens")
        self.assertEqual(note, "", "CAPTCHA frames take the pre-fix path: focus moving into the iframe counts as a reaction, no report")

    def test_frame_that_throws_is_skipped_not_fatal(self):
        page, child = _page_with_iframe(physical_click_works=False, dom_click_works=True)

        class Boom(FakeFrame):
            def frame_element(self):
                raise RuntimeError("Frame was detached")
        page.frames.append(Boom(parent=page.frames[0], box={"x": 0, "y": 0, "width": 1, "height": 1}))
        note = _click(page, IN_FRAME)           # the broken frame is skipped, the good one gets the fallback
        self.assertEqual(child.dom_clicks, [(100, 50)])
        self.assertEqual(note, "")


class TestHoverAndChartTooltips(_Base):
    """A tooltip changing from "Health services: 467" to "...: 502": same length, same node count; only the body-text
    hash changes. In one task most of 22 no-op notes hit steps like this, and the model had to fight through on screenshots alone."""

    def _hover(self, page, xy):
        wc.excute_action(page, FakeClient(page), {"name": "Hover", "coordinate": list(xy)})
        return wc.take_action_note()

    def test_hover_that_only_changes_text_hash_is_not_a_stall(self):
        page, child = _page_with_iframe()
        page.hover_changes_main_text = True
        self.assertEqual(self._hover(page, OUTSIDE), "")
        self.assertEqual(wc.no_change_streak(), 0)

    def test_hover_tooltip_inside_iframe_is_not_a_stall(self):
        page, child = _page_with_iframe(hover_changes_text=True)
        self.assertEqual(self._hover(page, IN_FRAME), "")
        self.assertEqual(child.dom_hovers, [], "the physical hover already showed the tooltip; no DOM events needed")

    def test_swallowed_hover_into_iframe_falls_back_to_dom_move(self):
        """An iframe at the bottom of the viewport does not even get mouse moves -> dispatch mousemove inside the child frame and the tooltip appears."""
        page, child = _page_with_iframe(hover_changes_text=False)
        self.assertEqual(self._hover(page, IN_FRAME), "")
        self.assertEqual(child.dom_hovers, [(100, 50)])

    def test_hover_fallback_that_also_fails_reports(self):
        page, child = _page_with_iframe(hover_changes_text=False)
        child.dom_hover_works = False
        note = self._hover(page, IN_FRAME)
        self.assertEqual(len(child.dom_hovers), 1)
        self.assertIn("Hovering there produced NO detectable change", note)
        self.assertIn("embedded frame", note)

    def test_hover_over_dead_space_still_reports(self):
        page, child = _page_with_iframe()
        note = self._hover(page, OUTSIDE)
        self.assertIn("Hovering there produced NO detectable change", note)

    def test_click_on_svg_bar_uses_text_hash_too(self):
        """Clicking a bar to read its tooltip: the target is a <path>, so chart rules apply."""
        page, child = _page_with_iframe()
        page.frames[0].tag = "path"               # the target is an SVG bar in the main frame
        page.mouse.click = lambda x, y, **kw: setattr(page, "fingerprint", dict(page.fingerprint, thash=7))
        self.assertEqual(_click(page, OUTSIDE), "")

    def test_click_on_ordinary_element_ignores_text_hash(self):
        """Ordinary elements ignore the hash: on a page with a clock the body text changes every second, and using it would switch off no-op detection."""
        page, child = _page_with_iframe()
        page.frames[0].tag = "DIV"
        page.mouse.click = lambda x, y, **kw: setattr(page, "fingerprint", dict(page.fingerprint, thash=7))
        self.assertIn("NO detectable change", _click(page, OUTSIDE))


class TestDescribePoint(_Base):
    def test_describes_element_inside_the_iframe(self):
        """Before the fix this only returned <IFRAME> and told the model it was not clickable."""
        page, child = _page_with_iframe()
        info = wc.describe_point(page, *IN_FRAME)
        self.assertEqual(info["tag"], "SPAN")
        self.assertTrue(info.get("frame"))

    def test_main_frame_point_has_no_frame_flag(self):
        page, child = _page_with_iframe()
        info = wc.describe_point(page, *OUTSIDE)
        self.assertEqual(info["tag"], "SPAN")     # the stand-in's main frame returns the same description
        self.assertFalse(info.get("frame"))


class TestConfig(unittest.TestCase):
    def test_switch_is_on_by_default_and_env_can_turn_it_off(self):
        self.assertTrue(RuntimeConfig().enable_iframe_click_fallback)
        cfg = RuntimeConfig.load(config_path="/nonexistent.json",
                                 env={"WEBFOVEA_ENABLE_IFRAME_CLICK_FALLBACK": "0"})
        self.assertFalse(cfg.enable_iframe_click_fallback)


if __name__ == "__main__":
    unittest.main()
