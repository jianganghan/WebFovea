"""Start src/agent/main.py with the same conditions on any machine.

The agent code keeps the logic it was evaluated with, so two machine differences
are bridged here rather than in src/:

1. Process start method. main.py defines its worker function inside the
   ``if __name__ == "__main__"`` block. Linux forks, so that works; macOS
   defaults to spawn, which re-imports the module and cannot find the function.
   We force fork.

2. Viewport. The agent was tuned with a 1920x1080 browser window, and
   open_page() sizes the viewport to the *window*. On a laptop the window gets
   clamped to the screen (e.g. 1440x875), so every screenshot and every
   coordinate would differ from those conditions. We override the viewport through
   Playwright's own set_viewport_size; a raw CDP Emulation call from a side
   session gets silently overwritten by Playwright's emulation state.
   Set WEBFOVEA_VIEWPORT=off to keep the window size, or e.g. 1600x900 to change it.

All command-line arguments are passed through to main.py unchanged.
"""
import multiprocessing as mp
import os
import runpy
import sys
from pathlib import Path

mp.set_start_method("fork", force=True)

MAIN_PY = Path(__file__).resolve().parent.parent / "src" / "agent" / "main.py"
sys.path.insert(0, str(MAIN_PY.parent))

_vp = os.environ.get("WEBFOVEA_VIEWPORT", "1920x1080")
if _vp.lower() not in ("off", "0", "no", ""):
    _w, _h = (int(x) for x in _vp.lower().split("x"))
    import web_controller as _wc
    from playwright.sync_api._generated import Page as _PWPage

    def _force_viewport(page):
        """Idempotent: does nothing if the viewport is already the target size."""
        try:
            vs = page.viewport_size
            if vs and vs.get("width") == _w and vs.get("height") == _h:
                return
            page.set_viewport_size({"width": _w, "height": _h})
            # web_controller keeps the window size in module globals and uses
            # them for scrolling and coordinate conversion; keep them in sync.
            _wc.browser_window_size_width = _w
            _wc.browser_window_size_height = _h
        except Exception as e:
            print(f"⚠️ viewport override failed, keeping the window size: {e}", flush=True)

    _orig_open_page = _wc.open_page

    def _open_page_forced_viewport(*args, **kwargs):
        page = _orig_open_page(*args, **kwargs)
        if page is not None:
            _force_viewport(page)
        return page

    _wc.open_page = _open_page_forced_viewport

    # Also before every screenshot: a page opened mid-task (e.g. a new tab)
    # never went through open_page().
    _orig_screenshot = _PWPage.screenshot

    def _screenshot_forced(self, **kwargs):
        _force_viewport(self)
        return _orig_screenshot(self, **kwargs)

    _PWPage.screenshot = _screenshot_forced

sys.argv[0] = str(MAIN_PY)
runpy.run_path(str(MAIN_PY), run_name="__main__")
