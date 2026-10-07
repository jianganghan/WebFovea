"""Task isolation of capture.json.

`capture.json` is the per-task record of how the answer was reached, and the standard is
**exactly this task's requests, no more and no less**. Workers are long-lived and one
browser context runs a dozen tasks in a row, so "no more" does not hold by itself; two layers
guarantee it:

  layer 1  RequestCollector.clear()  -- at the start of each task, unbind the listeners on all of the previous task's pages
  layer 2  close_other_pages()       -- at the start of each task, close the leftover pages themselves

Every test below targets a specific failure path; none is written for coverage's sake.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))
sys.path.insert(0, os.path.dirname(__file__))

from fakes import FakeContext, FakePage, FakeRequest      # noqa: E402
from web_controller import RequestCollector, close_other_pages  # noqa: E402
import main  # noqa: E402


class TestCrossTaskIsolation(unittest.TestCase):
    """Layer 1: unbinding listeners."""

    def setUp(self):
        self.collector = RequestCollector()

    def test_previous_task_page_does_not_leak_into_this_task(self):
        """Core regression: requests from a page left over by the previous task must never enter this task's capture.json.

        In the reference implementation the collector is a long-lived, per-worker object, while
        `page.on(...)` hangs on the page and pages survive across tasks, so the previous site's
        background polling was recorded in this task. Requests to another site (especially a search
        engine) in a task's record would be impossible to explain.
        """
        task1_page = FakePage("task1")
        self.collector.attach(task1_page)
        task1_page.emit_request(FakeRequest("https://site-a.example/api/data"))
        self.assertEqual(len(self.collector.requests), 1)

        # -- second task starts --
        self.collector.clear()
        task2_page = FakePage("task2")
        self.collector.attach(task2_page)

        # the previous task's page is still alive and keeps polling
        task1_page.emit_request(FakeRequest("https://site-a.example/api/poll"))
        task2_page.emit_request(FakeRequest("https://site-b.example/api/search"))

        urls = [r["url"] for r in self.collector.requests]
        self.assertEqual(urls, ["https://site-b.example/api/search"],
                         "the previous task's requests leaked into this task's capture.json")

    def test_clear_leaves_no_dangling_listener(self):
        page = FakePage("task1")
        self.collector.attach(page)
        self.assertEqual(page.listener_count("request"), 1)
        self.collector.clear()
        self.assertEqual(page.listener_count("request"), 0)
        self.assertEqual(self.collector.requests, [])

    def test_new_tab_opened_during_task_is_collected(self):
        """The "no less" side: requests from a tab opened during the task must be recorded too.

        When the main loop sees a page switch, it attaches to the new page. The answer is often on
        that new tab; missing it would leave capture.json incomplete.
        """
        first = FakePage("first")
        self.collector.attach(first)
        popup = FakePage("popup")            # opened by clicking a target=_blank link
        self.collector.attach(popup)

        popup.emit_request(FakeRequest("https://site.example/api/detail"))
        self.assertEqual([r["url"] for r in self.collector.requests],
                         ["https://site.example/api/detail"])

    def test_attach_is_idempotent(self):
        """Switching back and forth between tabs attaches to the same page repeatedly; without de-duplication each request is recorded twice."""
        page = FakePage("p")
        self.collector.attach(page)
        self.collector.attach(page)
        self.assertEqual(page.listener_count("request"), 1)

        page.emit_request(FakeRequest("https://site.example/api/x"))
        self.assertEqual(len(self.collector.requests), 1)

    def test_detach_survives_a_closed_page(self):
        """remove_listener raises on a closed page; a failed unbind must not bring down the task."""
        alive, dead = FakePage("alive"), FakePage("dead")
        self.collector.attach(alive)
        self.collector.attach(dead)
        dead.closed = True                    # the browser already closed it

        self.collector.clear()                # must not raise
        self.assertEqual(alive.listener_count("request"), 0)

    def test_only_xhr_and_fetch_are_recorded(self):
        """Regression guard: only data requests matter; images and stylesheets do not belong in capture.json."""
        page = FakePage("p")
        self.collector.attach(page)
        for rtype in ("document", "image", "stylesheet", "script", "font"):
            page.emit_request(FakeRequest(f"https://site.example/x.{rtype}", rtype))
        page.emit_request(FakeRequest("https://site.example/api/a", "xhr"))
        page.emit_request(FakeRequest("https://site.example/api/b", "fetch"))

        self.assertEqual([r["url"] for r in self.collector.requests],
                         ["https://site.example/api/a", "https://site.example/api/b"])


class TestCaptureIsMandatoryOutput(unittest.TestCase):
    """How capture.json gets written.

    It is the record of how a task was solved: it shows whether the agent operated the site or
    went through search engines / data endpoints.

    The reference implementation wrote it only on the **normal path**; the exception fallback
    wrote result.json alone, and tasks that took that path were missing their record.
    It is now written in exactly one place, `finally`; the three rules below pin that down.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.collector = RequestCollector()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def capture_path(self):
        return os.path.join(self.tmp, "capture.json")

    def test_written_even_when_the_task_blew_up(self):
        """Core regression: the exception path must leave a capture.json too."""
        page = FakePage("p1")
        self.collector.clear()
        self.collector.attach(page)
        page.emit_request(FakeRequest("https://example.com/api/x"))

        self.assertTrue(main.save_capture_json(self.collector, self.tmp, armed=True))
        with open(self.capture_path(), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total_requests"], 1)
        self.assertEqual(data["all_requests"][0]["url"], "https://example.com/api/x")

    def test_not_written_when_the_collector_was_never_armed_for_this_task(self):
        """The page-open failure path never reaches clear(), so the buffer still holds the **previous task's** requests.

        Writing it there would pour the previous task's requests into this task's directory, the
        cross-task contamination that had already been fixed once.
        **No file is better than a wrong one**: the first is missing evidence, the second is false evidence.
        """
        stale = FakePage("leftover from the previous task")
        self.collector.attach(stale)
        stale.emit_request(FakeRequest("https://previous-task.com/api/leak"))

        self.assertFalse(main.save_capture_json(self.collector, self.tmp, armed=False))
        self.assertFalse(os.path.exists(self.capture_path()),
                         "capture.json was written before the recorder was ready; its content belongs to the previous task")

    def test_never_raises_from_the_finally_block(self):
        """It is called from finally, in exactly the situations where something has already gone wrong.

        Raising from finally would replace the original exception and take the whole task loop down:
        one task failing to save would turn into a failure of every remaining task on this worker.
        """
        class Exploding:
            def save_results(self, path):
                raise OSError("disk full")

        self.assertFalse(
            main.save_capture_json(Exploding(), "/nonexistent/dir", armed=True))

    def test_reports_whether_it_actually_wrote(self):
        """The return value must distinguish "written" from "not written", or the caller cannot keep track."""
        self.assertTrue(main.save_capture_json(self.collector, self.tmp, armed=True))
        self.assertFalse(main.save_capture_json(self.collector, self.tmp, armed=False))


class TestSingleTabReset(unittest.TestCase):
    """Layer 2: single-tab reset."""

    def test_keeps_only_the_new_page(self):
        ctx = FakeContext()
        leftovers = [FakePage(f"old{i}", ctx) for i in range(3)]
        fresh = FakePage("fresh", ctx)

        closed = close_other_pages(ctx, keep=fresh)

        self.assertEqual(closed, 3)
        self.assertEqual(ctx.pages, [fresh])
        self.assertTrue(all(p.closed for p in leftovers))
        self.assertFalse(fresh.closed)

    def test_stops_in_flight_requests_before_closing(self):
        """window.stop() before close: otherwise in-flight requests fire another round of callbacks."""
        ctx = FakeContext()
        old = FakePage("old", ctx)
        fresh = FakePage("fresh", ctx)
        close_other_pages(ctx, keep=fresh)
        self.assertTrue(old.stopped)

    def test_one_stubborn_page_does_not_abort_the_rest(self):
        """A page that cannot be closed (dialog, crashed renderer) only means one extra tab stays open;
        it must not keep the following pages open too, let alone fail the task."""
        ctx = FakeContext()
        stubborn = FakePage("stubborn", ctx, close_error="target crashed")
        normal = FakePage("normal", ctx)
        fresh = FakePage("fresh", ctx)

        closed = close_other_pages(ctx, keep=fresh)

        self.assertEqual(closed, 1)
        self.assertTrue(normal.closed)
        self.assertIn(stubborn, ctx.pages)
        self.assertIn(fresh, ctx.pages)

    def test_isolation_still_holds_with_the_reset_switched_off(self):
        """What the kill switch (single_tab_reset=false) relies on: **with this layer off, cross-task
        isolation still holds**, because unbinding listeners in RequestCollector guarantees it independently.

        This test is that premise. If it ever fails, the kill switch is no longer a cheap fallback; it
        would bring capture.json contamination back.
        """
        collector = RequestCollector()
        ctx = FakeContext()
        task1 = FakePage("task1", ctx)
        collector.attach(task1)
        task1.emit_request(FakeRequest("https://site-a.example/api/x"))

        # second task starts, **without** the single-tab reset: the previous page stays in the context
        collector.clear()
        task2 = FakePage("task2", ctx)
        collector.attach(task2)
        self.assertIn(task1, ctx.pages, "this test assumes the old page was not closed")

        task1.emit_request(FakeRequest("https://site-a.example/api/poll"))
        task2.emit_request(FakeRequest("https://site-b.example/api/search"))

        self.assertEqual([r["url"] for r in collector.requests],
                         ["https://site-b.example/api/search"])

    def test_never_leaves_the_context_empty(self):
        """Dropping to 0 pages closes a CDP-connected browser too, and the whole worker is done for.
        That is why the new page is created before the old ones are closed, and why this assertion exists."""
        ctx = FakeContext()
        for i in range(4):
            FakePage(f"old{i}", ctx)
        keep = ctx.pages[-1]

        close_other_pages(ctx, keep=keep)

        self.assertEqual(ctx.pages, [keep])
        self.assertGreaterEqual(len(ctx.pages), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
