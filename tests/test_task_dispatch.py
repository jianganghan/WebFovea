"""Task dispatch: load balancing + worker crash compensation + blast-radius boundaries.

Why this layer deserves tests: every one of its failure modes is **"a few tasks went missing"**, and a
run that only reports a total score gives no per-task information. A mistake here is **invisible**
on the day; it only shows up as a slightly lower score. That is exactly how the static-sharding bug
behaved: when browser initialization failed the worker simply returned, its whole shard produced no
result.json, and the log had a single line saying "browser initialization failed, exiting".

These timings (who claims what, the moment of a crash, who picks it up) are nearly impossible to
reproduce in real runs, which cost real money and still might not trigger them. Here they are
deterministic, offline and free.

"A failed task scores zero but does not affect other tasks" is a **per-task** promise; the code must
really keep it per task and not degrade it to per worker.
"""
import multiprocessing as mp
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

from task_dispatch import TaskDispatcher, supervise_workers  # noqa: E402


class TestSharedPool(unittest.TestCase):
    """A shared pool instead of static shards: whoever is free claims, and each task goes to exactly one worker."""

    def test_tasks_are_handed_out_once_each(self):
        d = TaskDispatcher([1, 2, 3])
        seen = []
        for wid in (0, 1, 0):
            idx = d.claim(wid)
            seen.append(idx)
            d.complete(wid)
        self.assertEqual(sorted(seen), [1, 2, 3])
        self.assertIsNone(d.claim(0))      # pool empty = the worker should stop

    def test_fast_worker_takes_more_than_static_split_would_give(self):
        """Load balancing: static shards split 6 tasks across 2 workers as 3/3; a shared pool allows 5/1.

        That is the long-tail problem: total run time is the binding constraint.
        """
        d = TaskDispatcher(range(6))
        fast, slow = 0, 0
        slow_taken = d.claim(1)            # the slow worker claims one task and holds it
        while True:
            idx = d.claim(0)
            if idx is None:
                break
            fast += 1
            d.complete(0)
        self.assertEqual(fast, 5)
        self.assertIsNotNone(slow_taken)

    def test_claim_is_atomic_under_concurrency(self):
        """Concurrent claims must never hand the same task to two workers (the two trajectories would overwrite each other)."""
        d = TaskDispatcher(range(200))
        got = []
        lock = threading.Lock()

        def drain(wid):
            while True:
                idx = d.claim(wid)
                if idx is None:
                    return
                with lock:
                    got.append(idx)
                d.complete(wid)

        threads = [threading.Thread(target=drain, args=(w,)) for w in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(got), list(range(200)))   # none duplicated, none missing


class TestCrashCompensation(unittest.TestCase):
    """When a worker crashes, the task it held must go back to the pool; that is the main reason for the redesign."""

    def test_task_held_by_dead_worker_goes_back_to_pool(self):
        d = TaskDispatcher([7, 8])
        self.assertEqual(d.claim(0), 7)
        # The parent finds W0's process gone while it still holds a task
        self.assertEqual(d.release(0, reason="worker 0 process exited"), 7)
        # Another worker eventually gets it (back of the queue, so 8 comes first)
        self.assertEqual(d.claim(1), 8)
        d.complete(1)
        self.assertEqual(d.claim(1), 7)

    def test_requeued_task_goes_to_the_back_not_the_front(self):
        """Back of the queue: finish the healthy tasks first and defer known-risky ones to the end.

        At the front, the next idle worker would step on the same mine right away.
        """
        d = TaskDispatcher([1, 2, 3])
        self.assertEqual(d.claim(0), 1)
        d.release(0, reason="crashed")
        self.assertEqual([d.claim(1), d.claim(2), d.claim(3)], [2, 3, 1])

    def test_finished_task_is_never_re_dispatched(self):
        """A task that has written result.json is never re-sent.

        This is the regression the `finally: dispatcher.complete()` comment in main.py guards against:
        without complete, a worker exiting normally is mistaken for a mid-task crash, and a finished task
        is sent out again, overwriting its trajectory and burning the budget a second time.
        """
        d = TaskDispatcher([5])
        self.assertEqual(d.claim(0), 5)
        d.complete(0)                       # result saved
        self.assertIsNone(d.release(0, reason="worker 0 process exited"))
        self.assertEqual(d.pending_count(), 0)
        self.assertEqual(d.abandoned(), [])   # and it must not be recorded as abandoned either

    def test_release_on_idle_worker_is_a_noop(self):
        """Releasing a worker that stopped normally (holding nothing) must be harmless and must not invent a task."""
        d = TaskDispatcher([1])
        self.assertIsNone(d.release(3, reason="worker 3 exited normally"))
        self.assertEqual(d.pending_count(), 1)


class TestBlastRadiusLimits(unittest.TestCase):
    """A shared queue widens the blast radius from one shard to the whole pool; the boundaries are essential."""

    def test_poison_task_stops_bouncing_at_max_attempts(self):
        """Poison task: it takes down every worker that runs it, and without a cap it would bounce between all of them forever."""
        d = TaskDispatcher([9], max_attempts=2)
        self.assertEqual(d.claim(0), 9)
        self.assertEqual(d.release(0, reason="crashed"), 9)     # first crash: compensate
        self.assertEqual(d.claim(1), 9)
        self.assertIsNone(d.release(1, reason="crashed again"))      # second crash: give up
        self.assertEqual(d.pending_count(), 0)
        self.assertIsNone(d.claim(2))                       # never handed to anyone again
        self.assertEqual(d.abandoned(), [(9, "crashed again")])

    def test_default_config_forbids_any_task_level_retry(self):
        """A task that has started must never be retried or reset as a whole. The default must be 1."""
        from runtime_config import RuntimeConfig
        import os
        cfg = RuntimeConfig.load(config_path=os.path.join(
            os.path.dirname(__file__), "..", "config.json"), env={})
        self.assertEqual(cfg.max_task_attempts, 1,
                         "max_task_attempts > 1 would re-run a crashed task from step 0, i.e. reset the task")

    def test_max_attempts_1_means_a_crashed_task_is_never_rerun(self):
        """A crash is not compensated: the task is recorded as abandoned and never re-run.
        That is the intended behaviour, and its cost (losing that task) has to be accepted."""
        d = TaskDispatcher([4], max_attempts=1)
        self.assertEqual(d.claim(0), 4)
        self.assertIsNone(d.release(0, reason="crashed"))
        self.assertEqual(d.pending_count(), 0)

    def test_attempts_are_counted_per_task(self):
        d = TaskDispatcher([1, 2], max_attempts=3)
        d.claim(0); d.release(0, reason="crashed")       # 1 has been claimed once
        d.claim(0)                                  # claims 2
        self.assertEqual(d.attempts_of(1), 1)
        self.assertEqual(d.attempts_of(2), 1)
        self.assertEqual(d.attempts_of(999), 0)     # unknown tasks do not raise


class TestNothingVanishesSilently(unittest.TestCase):
    """Every missing task must leave a record. Silently dropped tasks are the hardest failure to notice afterwards."""

    def test_drain_pending_records_every_leftover(self):
        d = TaskDispatcher([1, 2, 3])
        d.claim(0)
        d.complete(0)
        left = d.drain_pending("all workers exited; nobody claimed it")
        self.assertEqual(sorted(left), [2, 3])
        self.assertEqual(d.pending_count(), 0)
        self.assertEqual(
            sorted(d.abandoned()),
            [(2, "all workers exited; nobody claimed it"), (3, "all workers exited; nobody claimed it")])

    def test_abandoned_keeps_the_reason(self):
        """The reason string goes into summary.json; when only a total score comes back, it is the only clue."""
        d = TaskDispatcher([1], max_attempts=1)
        d.claim(0)
        d.release(0, reason="page initialization failed: TimeoutError")
        self.assertEqual(d.abandoned(), [(1, "page initialization failed: TimeoutError")])


class TestWallClockGate(unittest.TestCase):
    """Wall-clock gate. Total run time is the real constraint, and crash compensation and worker restarts both
    lengthen it; without a gate the redesign would reintroduce the problem along the very axis it protects.

    Fake clock, no sleep: time-dependent tests that wait for real are slow and flaky.
    """

    def setUp(self):
        self.now = 1000.0

    def clock(self):
        return self.now

    def test_no_new_tasks_after_deadline(self):
        d = TaskDispatcher([1, 2, 3], deadline=1010.0, time_fn=self.clock)
        self.assertEqual(d.claim(0), 1)
        d.complete(0)
        self.now = 1010.0
        self.assertIsNone(d.claim(0))
        self.assertTrue(d.expired())
        self.assertEqual(d.pending_count(), 2, "the gate only blocks dispatch; it must not quietly empty the pool")

    def test_running_task_is_not_interrupted(self):
        """Past the deadline only new tasks are blocked. Killing a running task wastes the steps spent and loses its result.json."""
        d = TaskDispatcher([1, 2], deadline=1010.0, time_fn=self.clock)
        self.assertEqual(d.claim(0), 1)
        self.now = 1020.0
        self.assertEqual(d.inflight_of(0), 1)      # still assigned to it
        d.complete(0)                               # finishes and is settled as usual
        self.assertEqual(d.abandoned(), [])

    def test_leftovers_are_recorded_with_the_deadline_reason(self):
        d = TaskDispatcher([1, 2], deadline=1010.0, time_fn=self.clock)
        self.now = 1010.0
        left = d.drain_pending("wall-clock gate closed before it started")
        self.assertEqual(sorted(left), [1, 2])
        self.assertEqual([r for _, r in d.abandoned()],
                         ["wall-clock gate closed before it started"] * 2)

    def test_gate_off_by_default(self):
        d = TaskDispatcher([1])
        self.assertFalse(d.expired())
        self.assertEqual(d.claim(0), 1)


class FakeProc:
    """Worker process stand-in. The number of polls it survives is set in advance, which makes crash timing deterministic."""

    def __init__(self, alive_polls):
        self.alive_polls = alive_polls
        self.terminated = False
        self.killed = False
        self.joined = 0

    def is_alive(self):
        if self.alive_polls <= 0:
            return False
        self.alive_polls -= 1
        return True

    def terminate(self):
        self.terminated = True
        self.alive_polls = 0

    def join(self, timeout=None):
        self.joined += 1

    def kill(self):
        self.killed = True


class TestSupervision(unittest.TestCase):
    """The parent's reaping / compensation / restarts. The most branch-heavy part, and when it goes wrong it too only shows up as "a few tasks went missing"."""

    def setUp(self):
        self.started = []       # order in which start_worker was called

    def make_starter(self, plan):
        """plan: worker_id -> [polls survived on 1st start, on 2nd start, ...]"""
        counts = {}

        def start(wid):
            n = counts.get(wid, 0)
            counts[wid] = n + 1
            self.started.append(wid)
            lifetimes = plan.get(wid, [0])
            return FakeProc(lifetimes[min(n, len(lifetimes) - 1)])
        return start

    def run_supervisor(self, dispatcher, worker_ids, plan, **kw):
        kw.setdefault("stagger_s", 0)
        kw.setdefault("poll_interval", 0)
        kw.setdefault("sleep_fn", lambda _: None)
        kw.setdefault("log", lambda *a, **k: None)
        return supervise_workers(dispatcher, worker_ids,
                                 self.make_starter(plan), **kw)

    def test_dead_worker_task_is_requeued_and_another_worker_gets_it(self):
        """The main reason for the redesign: when a worker crashes mid-task, the task must not disappear with it."""
        d = TaskDispatcher([10, 11])
        d.claim(0)                                  # W0 holds 10 and crashes
        self.run_supervisor(d, [0], {0: [0]}, max_respawns=0)
        self.assertEqual(d.pending_count(), 2)      # 10 is back in the pool, 11 not yet handed out
        self.assertEqual(d.abandoned(), [])
        self.assertIn(10, [d.claim(1), d.claim(1)])

    def test_normal_exit_is_not_treated_as_a_crash(self):
        """A worker that stopped normally holds nothing; nothing should be invented for it."""
        d = TaskDispatcher([1])
        d.claim(0)
        d.complete(0)                               # result saved, then a normal exit
        self.run_supervisor(d, [0], {0: [0]}, max_respawns=0)
        self.assertEqual(d.pending_count(), 0)
        self.assertEqual(d.abandoned(), [])

    def test_worker_is_respawned_while_work_remains_but_capped(self):
        """If there is work left, restore the slot (losing one of N workers costs 1/N of the throughput
        against a fixed time budget), but with a cap: a slot whose CDP address is gone for good would
        loop through start -> init failure -> exit forever."""
        d = TaskDispatcher(range(5))
        respawns = self.run_supervisor(d, [0], {0: [0]}, max_respawns=2)
        self.assertEqual(respawns.get(0), 2)
        self.assertEqual(self.started.count(0), 3)   # first start + 2 restarts, then let go

    def test_no_respawn_when_the_pool_is_empty(self):
        """Once all the work is done, do not start more processes."""
        d = TaskDispatcher([1])
        d.claim(0)
        d.complete(0)
        respawns = self.run_supervisor(d, [0], {0: [0]}, max_respawns=2)
        self.assertEqual(respawns, {})
        self.assertEqual(self.started.count(0), 1)

    def test_no_respawn_after_the_wall_clock_gate(self):
        """Past the deadline, wind down and start no new processes; that is exactly what the gate is for."""
        d = TaskDispatcher(range(5), deadline=0.0, time_fn=lambda: 1.0)
        respawns = self.run_supervisor(d, [0], {0: [0]}, max_respawns=2)
        self.assertEqual(respawns, {})
        self.assertEqual(self.started.count(0), 1)

    def test_poison_task_is_abandoned_not_bounced_forever(self):
        """Once a poison task has used up both claims it must be recorded, not keep bouncing between workers."""
        d = TaskDispatcher([42], max_attempts=2)
        d.claim(0)
        self.run_supervisor(d, [0], {0: [0]}, max_respawns=0)   # first crash: compensate
        self.assertEqual(d.pending_count(), 1)
        d.claim(1)
        self.run_supervisor(d, [1], {1: [0]}, max_respawns=0)   # second crash: give up
        self.assertEqual(d.pending_count(), 0)
        self.assertEqual(d.abandoned(), [(42, "worker 1 process exited")])


class TestManagerProxyBackend(unittest.TestCase):
    """Production uses Manager proxies, not plain list/dict.

    Tested separately because a proxy's methods are **not the same** as the built-in types': does
    ListProxy support `pop(0)`, do DictProxy's `pop(k, default)` and `get` work? Only a test can tell.
    If this layer is wrong, every worker raises at once and the run never gets going.
    """

    def test_works_over_multiprocessing_manager(self):
        with mp.Manager() as manager:
            d = TaskDispatcher(
                [1, 2, 3],
                max_attempts=2,
                list_factory=manager.list,
                dict_factory=manager.dict,
                lock=manager.Lock(),
            )
            self.assertEqual(d.total, 3)
            self.assertEqual(d.claim(0), 1)
            self.assertEqual(d.inflight_of(0), 1)
            self.assertEqual(d.inflight_count(), 1)
            self.assertEqual(d.release(0, reason="crashed"), 1)
            self.assertEqual(d.attempts_of(1), 1)
            self.assertEqual(d.pending_count(), 3)
            d.complete(0)                     # a no-op, must not raise
            self.assertEqual(sorted(d.drain_pending("wrap-up")), [1, 2, 3])
            self.assertEqual(len(d.abandoned()), 3)


if __name__ == "__main__":
    unittest.main()
