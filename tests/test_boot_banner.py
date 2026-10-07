"""Boot banner: the one thing that gets through when the agent runs on someone else's machine.

Once a run is going, step-by-step logs land in `{output}/logs/worker_N_*.log`. But
**a crash before startup leaves no trace at all**: dependencies missing, import
failure, broken config; no worker started, logs/ is empty, and only run.sh's stdout remains.

So the requirements on these lines are specific: they must be **printed first**, must
**never raise**, and must carry an unmistakable `BOOT OK` marker. Its absence means
"startup never completed", and knowing that is worth a lot on its own.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

import main  # noqa: E402


def render():
    buf = io.StringIO()
    with redirect_stdout(buf):
        main.print_boot_banner("/out", "/out/logs",
                               ["http://cdp1", "http://cdp2"], 100,
                               "claude-opus-4-6", "https://api.anthropic.com/v1")
    return buf.getvalue()


class TestBootBanner(unittest.TestCase):

    def test_carries_everything_needed_to_identify_the_run(self):
        """Beyond "it failed", these are the things we want to know: which model, which
        parameters, the right number of tasks and browsers, and where the output went."""
        out = render()
        for expected in ("claude-opus-4-6",          # right model?
                         "api.anthropic.com",         # right endpoint?
                         "max_steps=80",              # the parameters we think we are running?
                         "single_tab_reset",          # was the kill switch on or off?
                         "tasks            : 100",    # how many tasks were read from the file
                         "workers (cdp)    : 2",      # browsers = workers
                         "/out/logs"):                # where the logs went
            self.assertIn(expected, out, f"banner is missing {expected!r}")

    def test_prints_an_unmistakable_success_marker(self):
        """The value of `BOOT OK` is in its **absence**: if the returned stdout lacks this line,
        startup never finished, which rules out every theory about the run itself."""
        self.assertIn("BOOT OK", render())

    def test_reports_the_playwright_version(self):
        """A missing or wrong Playwright is the most common startup failure, and it decides
        whether newer APIs such as `select()` are available."""
        self.assertIn("playwright       :", render())

    def test_never_raises_even_if_a_dependency_is_missing(self):
        """The banner itself must never become a new startup crash; that would defeat its purpose."""
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __builtins__.__import__

        def broken_import(name, *a, **kw):
            if name == "playwright":
                raise ImportError("No module named 'playwright'")
            return real_import(name, *a, **kw)

        import builtins
        builtins.__import__ = broken_import
        try:
            out = render()          # must not raise
        finally:
            builtins.__import__ = real_import
        self.assertIn("IMPORT FAILED", out)
        self.assertIn("BOOT OK", out, "the banner must still finish when a dependency is missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
