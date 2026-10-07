"""Task dispatch across workers: a shared queue with crash compensation.

**Why not static sharding.** The reference implementation split the task list up
front with `np.array_split(indices, WORKERS)`, and each worker worked through its
own share. That has two consequences:

1. **One dead worker loses its whole share.** When browser initialization fails,
   `run_worker` simply returns, and the rest of its share never produces a
   `result.json`. "A failed task scores zero but does not affect other tasks" is a
   per-task promise; static sharding turns it into a per-worker one, so a single
   CDP hiccup costs a dozen tasks instead of one.
2. **The long tail sets the total run time.** A run has a fixed wall-clock budget,
   and the step limit is not the binding constraint (see the comment on
   `runtime_config.max_steps`). A worker that drew hard tasks keeps running while
   the one that drew easy tasks sits idle.

A shared queue fixes both: whoever is free takes the next task (load balancing),
and the task a dead worker was holding goes back to the pool for someone else
(crash compensation).

**Four boundaries.** They are not excess caution: a shared queue widens the blast
radius from one shard to the whole pool, and without limits we would be buying
load balancing with more risk.

- `max_attempts`: **keep it at 1. A task that has started is never re-run from
  scratch.** With a value above 1, this class does exactly that: a worker crashes
  mid-task, the task returns to the pool, another worker claims it,
  `safe_create_directory` deletes the trajectory, and the task starts again from
  step 0. That discards evidence and repeats whatever the first attempt did on the
  live site; many benchmark protocols also forbid restarting a task once it has
  begun. At 1, the task held by a crashed worker is recorded in `abandoned()`
  instead: recorded, not re-run. When only a total score comes back, "this task
  was abandoned" is the only clue there is.
  ⚠️ **Load balancing is unaffected**: handing a task that has *not started* to an
  idle worker is assignment, not a reset.
- `release` and `complete` are kept strictly apart. Only a task whose page never
  opened goes back to the pool; a task that has written `result.json` is **never
  dispatched again**. Re-running a finished task only burns budget and overwrites
  its trajectory.
- Released tasks go to the **back** of the queue, not the front. At the front, the
  next idle worker would step on the same mine right away; at the back, the healthy
  tasks get done first and known-risky work is pushed to the end. Under the same
  `max_attempts` cap, that finishes the most tasks before anything blows up.
- `deadline`: a wall-clock gate. Crash compensation and worker restarts both
  *lengthen* a run, and the total run time is exactly what this redesign protects;
  without a gate we would reintroduce the problem along the very axis we optimized.
  Once the gate closes it only blocks new tasks and never interrupts running ones:
  killing a task midway wastes the steps already spent and loses its `result.json`.

A fifth boundary lives on the worker side (`consecutive_page_failures` in
`main.py`): a worker whose browser is dead fails every task it claims within
seconds. Under static sharding it only ruined its own share; with a shared queue it
could eat everyone's tasks. So after enough consecutive page-open failures it exits
and leaves the pool to healthy workers.

**Concurrency model**: all state sits behind one lock, and `claim` / `complete` /
`release` are atomic. Production passes `multiprocessing.Manager` proxies (shared
across processes); tests pass the standard library's `list` / `dict` /
`threading.Lock` (no processes, offline, free). What is under test is the dispatch
policy itself, which has nothing to do with Playwright or the model, so this
module imports neither.
"""
from __future__ import annotations

import threading
import time


class TaskDispatcher:
    """Shared pool of tasks waiting to run.

    Args:
        indices: positions of the pending tasks in the task file, in original order.
        max_attempts: how many times a task may be claimed. **This is not "retry on
            failure"**: a failed task still scores zero and is never re-sent; the
            count only matters for compensating a worker crash. 1 means "no
            compensation", the old static-sharding behaviour, but with load balancing.
        deadline: wall-clock gate, as an absolute time on the same clock as
            `time_fn`. Past it, `claim` always returns None; running tasks are not
            affected. None means no limit.
        time_fn: returns the current time; `time.monotonic` by default. Tests inject
            a fake clock instead of sleeping.
        list_factory / dict_factory / lock: how the shared state is built. Standard
            library by default; production passes Manager proxies.
    """

    def __init__(self, indices, *, max_attempts: int = 2, deadline=None,
                 time_fn=time.monotonic,
                 list_factory=list, dict_factory=dict, lock=None):
        self._deadline = deadline
        self._time_fn = time_fn
        self._pending = list_factory(list(indices))
        self._inflight = dict_factory()       # worker_id -> task index
        self._attempts = dict_factory()       # task index -> times claimed
        self._abandoned = list_factory()      # [(task index, reason), ...]
        self._lock = threading.Lock() if lock is None else lock
        self._max_attempts = max_attempts
        self._total = len(self._pending)

    # ── worker side ────────────────────────────────────────────────────────
    def claim(self, worker_id):
        """Claim a task. None means the worker should stop (pool empty or gate closed).

        The gate only blocks **new** tasks and never interrupts a running one:
        killing a task midway wastes the steps already spent and loses its
        result.json.
        """
        with self._lock:
            if self._deadline is not None and self._time_fn() >= self._deadline:
                return None
            if len(self._pending) == 0:
                return None
            idx = self._pending.pop(0)
            self._attempts[idx] = self._attempts.get(idx, 0) + 1
            self._inflight[worker_id] = idx
            return idx

    def complete(self, worker_id):
        """The task has an outcome (success or failure) and is **never re-sent**.

        Call it from a finally block. If any exit path skips it, the worker keeps
        the previous task as inflight, and when the process later exits normally the
        parent mistakes it for a mid-task crash and re-sends a task that already has
        a result.json, which would then be wiped and re-run.
        """
        with self._lock:
            self._inflight.pop(worker_id, None)

    def release(self, worker_id, reason: str = "released"):
        """The task never got started (page did not open / worker process died): return it.

        Returns the released task index. If the task has reached `max_attempts`, it is
        recorded as abandoned and None is returned. Also returns None when the worker
        holds no task (a normal shutdown).
        """
        with self._lock:
            idx = self._inflight.pop(worker_id, None)
            if idx is None:
                return None
            if self._attempts.get(idx, 0) >= self._max_attempts:
                self._abandoned.append((idx, reason))
                return None
            self._pending.append(idx)   # back of the queue: defer known-risky work
            return idx

    def expired(self) -> bool:
        """Whether the wall-clock gate has closed. The parent uses it for its final
        report and to decide whether restarting a worker is still worth it."""
        return self._deadline is not None and self._time_fn() >= self._deadline

    # ── parent side ────────────────────────────────────────────────────────
    def drain_pending(self, reason: str):
        """No workers left but tasks remain: record them all as abandoned, return their indices.

        These tasks are a pure loss (not even a result.json) and must be printed in
        the final report: quietly missing tasks are the hardest failure to notice
        after the fact.
        """
        with self._lock:
            left = list(self._pending)
            for idx in left:
                self._abandoned.append((idx, reason))
            while len(self._pending):
                self._pending.pop()
            return left

    # ── read-only views ────────────────────────────────────────────────────
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def inflight_of(self, worker_id):
        with self._lock:
            return self._inflight.get(worker_id)

    def attempts_of(self, idx) -> int:
        with self._lock:
            return self._attempts.get(idx, 0)

    def abandoned(self):
        """[(task index, reason), ...] in the order they happened."""
        with self._lock:
            return [tuple(x) for x in self._abandoned]

    @property
    def total(self) -> int:
        return self._total


def supervise_workers(dispatcher, worker_ids, start_worker, *, max_respawns=2,
                      stagger_s=3.0, poll_interval=5.0,
                      sleep_fn=time.sleep, log=print, status_extra=None):
    """Start and supervise worker processes: reap, compensate, restart if needed.

    Returns {worker_id: number of restarts}.

    **Why this is pulled out of main() too.** It is the most branch-heavy part of
    the redesign (did the process finish normally or crash? if it crashed, does its
    task go back? should this slot be restarted?), and when it goes wrong it fails
    just like the dispatcher does: **a few tasks quietly go missing**. A run that
    only reports a total score gives no sign of that. Left inside main(), it would
    be tied to Playwright and the model and could only be exercised by luck in a
    real run.

    `start_worker(wid)` may return anything with `is_alive / terminate / join /
    kill`: production passes `multiprocessing.Process`, tests pass a stand-in, which
    makes timings such as "crashes on the third poll" assertable.

    Reaping is decided by **whether the worker still holds an inflight task**:
      yes -> it crashed mid-task, which left no result.json -> return the task;
      no  -> it stopped normally (pool empty) or quit after repeated page-open
             failures -> nothing to compensate.
    """
    procs = {}
    respawns = {}
    for wid in worker_ids:
        procs[wid] = start_worker(wid)
        log(f"Worker {wid} started")
        if stagger_s:
            sleep_fn(stagger_s)

    try:
        while procs:
            sleep_fn(poll_interval)
            for wid, proc in list(procs.items()):
                if proc.is_alive():
                    continue
                held = dispatcher.inflight_of(wid)      # read first; release() clears it
                requeued = dispatcher.release(wid, reason=f"worker {wid} process exited")
                if requeued is not None:
                    log(f"⚠️ Worker {wid} exited during task {requeued}; the task went back to the pool")
                elif held is not None:
                    log(f"🔴 Worker {wid} exited during task {held}, "
                        f"which has reached its claim limit; task abandoned")
                del procs[wid]

                # If there is work left, restore this slot: losing one of N workers
                # costs 1/N of the throughput against a fixed time budget. Restarts
                # must be capped, or a slot whose CDP address is gone for good loops
                # forever through start -> init failure -> exit -> restart.
                if dispatcher.pending_count() == 0 or dispatcher.expired():
                    continue
                used = respawns.get(wid, 0)
                if used >= max_respawns:
                    log(f"⚠️ Worker {wid} has been restarted {used} times; not restarting again")
                    continue
                respawns[wid] = used + 1
                log(f"↻ Restarting worker {wid} ({used + 1}/{max_respawns}), "
                    f"{dispatcher.pending_count()} task(s) still in the pool")
                procs[wid] = start_worker(wid)

            if procs:
                extra = f", {status_extra()}" if status_extra else ""
                log(f"[status] active: {len(procs)}, running: {dispatcher.inflight_count()}, "
                    f"queued: {dispatcher.pending_count()}{extra}")
    except KeyboardInterrupt:
        log("\nInterrupted, terminating all processes...")
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
    for proc in procs.values():
        proc.join()
    return respawns
