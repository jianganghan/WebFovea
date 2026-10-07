"""Settling after navigation: floor / early exit / ceiling, all three must hold.

It replaces the old `page.wait_for_timeout(3000)`, a blind bet that 3 seconds would do.
`domcontentloaded` only means the DOM skeleton exists, not that the content is there; many target
sites are SPAs / data portals that "download and run a bundle -> send a data XHR -> render once the
data arrives". In one development set 8 of 70 tasks kept complaining that the page had not loaded,
and 6 of them ended without an answer.

Each stage corresponds to a specific bad case, so each has its own test:
  no floor      -> waits *less* than the old fixed 3 seconds (the only real risk of this change)
  no early exit -> fast sites waste 2+ seconds for nothing
  no ceiling    -> pages with polling / beacons never reach networkidle and wait until the timeout
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

from fakes import FakePage  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402
from web_controller import settle_after_navigation  # noqa: E402

REPO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Small values on the real clock: the whole suite takes about 100 ms and still tests real timing
FLOOR, CEIL = 30, 120


class SettlingPage(FakePage):
    """Controllable load states: choose which load states are never reached."""

    def __init__(self, *a, never_settles=(), **kw):
        super().__init__(*a, **kw)
        self.never_settles = set(never_settles)
        self.awaited = []

    def wait_for_timeout(self, ms):
        super().wait_for_timeout(ms)
        time.sleep(ms / 1000.0)

    def wait_for_load_state(self, state=None, timeout=None):
        self.awaited.append(state)
        if state in self.never_settles:
            time.sleep((timeout or 0) / 1000.0)          # really wait out the timeout
            raise TimeoutError(f"Timeout waiting for load state {state!r}")


class TestSettle(unittest.TestCase):

    def test_floor_is_respected_even_when_everything_settles_instantly(self):
        """The only real risk: without a floor, a page that reaches networkidle first and draws content a moment
        later would be screenshotted blank, worse than the old fixed 3 seconds."""
        page = SettlingPage("p")
        waited = settle_after_navigation(page, floor_ms=FLOOR, ceil_ms=CEIL)
        self.assertGreaterEqual(waited, FLOOR)

    def test_returns_early_when_the_page_settles(self):
        """A fast site should not sit out the full ceiling; this is the time saved."""
        page = SettlingPage("p")
        waited = settle_after_navigation(page, floor_ms=FLOOR, ceil_ms=CEIL)
        self.assertLess(waited, CEIL, "the page was already quiet, yet the full ceiling was waited")
        self.assertEqual(page.awaited, ["load", "networkidle"])

    def test_ceiling_caps_a_page_that_never_goes_idle(self):
        """Pages with polling / beacons / WebSockets never reach "no connections for 500 ms".
        Without a ceiling this would wait until Playwright's default timeout."""
        page = SettlingPage("p", never_settles=["networkidle"])
        started = time.time()
        waited = settle_after_navigation(page, floor_ms=FLOOR, ceil_ms=CEIL)
        elapsed = (time.time() - started) * 1000
        self.assertLessEqual(waited, CEIL + 40, "exceeded the ceiling")
        self.assertLessEqual(elapsed, CEIL + 40)

    def test_gives_up_on_the_rest_once_a_state_times_out(self):
        """For a page that does not even reach `load`, waiting for networkidle too only eats the rest of the budget."""
        page = SettlingPage("p", never_settles=["load", "networkidle"])
        settle_after_navigation(page, floor_ms=FLOOR, ceil_ms=CEIL)
        self.assertEqual(page.awaited, ["load"], "after load times out, do not go on to wait for networkidle")

    def test_never_raises(self):
        """Not settling is not an error. Raising would bubble up to the main loop's except and cost the whole task."""
        page = SettlingPage("p", never_settles=["load"])
        settle_after_navigation(page, floor_ms=FLOOR, ceil_ms=CEIL)   # must not raise

    def test_defaults_come_from_config(self):
        """Both parameters must be configurable: if a run shows they are wrong, change them without touching code."""
        cfg = RuntimeConfig.load(config_path=REPO_CONFIG, env={})
        self.assertEqual(cfg.settle_floor_ms, 1000)
        self.assertEqual(cfg.settle_ceil_ms, 8000)
        self.assertGreater(cfg.settle_ceil_ms, cfg.settle_floor_ms,
                           "the ceiling must be above the floor, or the early exit means nothing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
