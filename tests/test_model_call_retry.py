"""Model-call retries: attempts and total time are configurable, and three 403s at step 0 no longer end a task. Offline, zero tokens.

**Root cause**: retries were fixed at 3 attempts with 2/4/8 s backoff, 14 s in total. One task got
three 403 "Request not allowed" responses in a row at step 0: predict() returned FAIL with no actions,
main.py took `get_actions()[-1]`, raised IndexError, and the task ended at step 0. 4 of 305 tasks died
this way (1.3%),
while these 403s are sporadic policy refusals: about 50 others in the same period all passed on the second attempt.

Two things to guard: 1. more attempts really do save the task; 2. the time cap stops the bad case where
every attempt hangs until the timeout, so one task cannot burn the wall-clock budget.
"""
import io
import os
import sys
import types
import unittest
import unittest.mock

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

import agent as agent_mod  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(buf, format="PNG")
    return buf.getvalue()


class FakeClock:
    """Stand-in for time.time / time.sleep: sleep advances the clock, and a call itself can take time (to simulate timeouts)."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class FakeCompletions:
    def __init__(self, clock, fail_times, status=403, call_cost_s=0.0):
        self.clock = clock
        self.fail_times = fail_times
        self.status = status
        self.call_cost_s = call_cost_s
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        self.clock.now += self.call_cost_s
        if self.calls <= self.fail_times:
            e = RuntimeError(f"Error code: {self.status}")
            e.status_code = self.status
            raise e
        msg = types.SimpleNamespace(content="Thought: click.\nAction: click(start_box='(100,100)')")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=None, model="fake")


def _run(fail_times, status=403, call_cost_s=0.0, **cfg_kw):
    clock = FakeClock()
    fake_time = types.SimpleNamespace(time=clock.time, sleep=clock.sleep)
    cfg = RuntimeConfig(**cfg_kw)
    with unittest.mock.patch.object(agent_mod, "time", fake_time), \
            unittest.mock.patch.object(agent_mod, "get_config", return_value=cfg):
        ag = agent_mod.UITARSAgent(model="fake", api_url="http://127.0.0.1:1/v1", api_key="x",
                                   infer_mode="qwen25vl_normal")
        comp = FakeCompletions(clock, fail_times, status, call_cost_s)
        ag.vlm = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
        _, condition = ag.predict("task", {"screenshot": _png()})
    return condition, comp.calls, clock.sleeps, ag


class TestRetry(unittest.TestCase):
    def test_four_403s_in_a_row_no_longer_kill_the_step(self):
        """With the old limit of 3, a 4th attempt that would succeed never happens."""
        condition, calls, sleeps, ag = _run(fail_times=4)
        self.assertEqual(condition, "CONTINUE")
        self.assertEqual(calls, 5)
        self.assertEqual(sleeps, [2, 4, 8, 16])
        self.assertEqual(len(ag.get_actions()), 1)

    def test_old_limit_of_three_reproduces_the_w29_death(self):
        """Setting the limit back to 3 reproduces the original failure, which shows the test can tell the difference."""
        condition, calls, _, ag = _run(fail_times=4, model_call_max_attempts=3)
        self.assertEqual(condition, "FAIL")
        self.assertEqual(calls, 3)
        self.assertEqual(ag.get_actions(), [], "a failed call must leave no action, or the main loop would execute it as real")

    def test_time_budget_stops_retrying_when_every_call_times_out(self):
        """Every call hangs for 120 s: within the 240 s budget there is room for one more attempt, then it gives up, so one task cannot burn the wall clock."""
        condition, calls, sleeps, _ = _run(fail_times=99, status=None, call_cost_s=120,
                                           model_call_retry_budget_s=240)
        self.assertEqual(condition, "FAIL")
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, [2])

    def test_backoff_is_capped_at_30s(self):
        _, _, sleeps, _ = _run(fail_times=6, model_call_max_attempts=7)
        self.assertEqual(sleeps, [2, 4, 8, 16, 30, 30])

    def test_non_retryable_4xx_fails_fast(self):
        condition, calls, sleeps, _ = _run(fail_times=99, status=400)
        self.assertEqual((condition, calls, sleeps), ("FAIL", 1, []))

    def test_defaults_and_env_override(self):
        cfg = RuntimeConfig()
        self.assertEqual((cfg.model_call_max_attempts, cfg.model_call_retry_budget_s), (6, 240))
        cfg = RuntimeConfig.load(config_path="/nonexistent.json",
                                 env={"WEBFOVEA_MODEL_CALL_MAX_ATTEMPTS": "2", "WEBFOVEA_MODEL_CALL_RETRY_BUDGET_S": "0"})
        self.assertEqual(cfg.model_call_max_attempts, 2)
        self.assertEqual(cfg.model_call_retry_budget_s, 240, "0 is not a valid budget; that source should be ignored")


if __name__ == "__main__":
    unittest.main()
