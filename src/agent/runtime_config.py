"""Single entry point for the agent's runtime parameters.

**Why a separate module** rather than `os.environ.get(...)` scattered across call sites:

1. A parameter defined in several places will eventually be changed in only one of
   them. `max_steps` once lived in two places in `main.py` plus
   `max_trajectory_length` in `agent.py`, and the smallest one won: we thought the
   limit was 100 while the agent actually stopped at 50. A parameter is only really
   "set" when it has a single definition.
2. The command line of an evaluation entry point is often fixed and environment
   variables can be awkward to inject, so the configuration must be able to live
   in a file.
3. **Why each value is what it is** needs a place to be written down. The
   reasoning matters more than the number.

**Priority: environment variable > config.json "agent" section > default in code.**

- Environment variables win, so a local diagnostic run can, say, raise the step
  limit without touching `config.json` (and without the risk of forgetting to
  change it back).
- Defaults in code are the last resort. `config.json` may be missing, replaced or
  rewritten; a missing "agent" section, a missing field or a field of the wrong
  type must all fall back to the default and keep going. **No configuration error
  may stop a run from starting**, which is why parsing below is wrapped in try and
  bad values only print a warning.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields

# Location of config.json relative to this file: src/agent/runtime_config.py -> <repo>/config.json
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config.json")

# Environment variable for each field.
_ENV_KEYS = {
    "max_steps": "WEBFOVEA_MAX_STEPS",
    "task_budget_s": "WEBFOVEA_TASK_BUDGET_S",
    "max_reconnect_per_task": "WEBFOVEA_MAX_RECONNECT_PER_TASK",
    "default_reference_length": "WEBFOVEA_DEFAULT_REFERENCE_LENGTH",
    "settle_floor_ms": "WEBFOVEA_SETTLE_FLOOR_MS",
    "settle_ceil_ms": "WEBFOVEA_SETTLE_CEIL_MS",
    "enable_iframe_click_fallback": "WEBFOVEA_ENABLE_IFRAME_CLICK_FALLBACK",
    "enable_provenance_rule": "WEBFOVEA_ENABLE_PROVENANCE_RULE",
    "enable_click_many": "WEBFOVEA_ENABLE_CLICK_MANY",
    "model_call_max_attempts": "WEBFOVEA_MODEL_CALL_MAX_ATTEMPTS",
    "model_call_retry_budget_s": "WEBFOVEA_MODEL_CALL_RETRY_BUDGET_S",
    "single_tab_reset": "WEBFOVEA_SINGLE_TAB_RESET",
    "max_task_attempts": "WEBFOVEA_MAX_TASK_ATTEMPTS",
    "max_worker_respawns": "WEBFOVEA_MAX_WORKER_RESPAWNS",
    "worker_init_retries": "WEBFOVEA_WORKER_INIT_RETRIES",
    "max_consecutive_page_failures": "WEBFOVEA_MAX_CONSECUTIVE_PAGE_FAILURES",
    "run_budget_s": "WEBFOVEA_RUN_BUDGET_S",
    "enable_back": "WEBFOVEA_ENABLE_BACK",
    "max_backs_per_task": "WEBFOVEA_MAX_BACKS_PER_TASK",
    "enable_find_text": "WEBFOVEA_ENABLE_FIND_TEXT",
    "max_find_text_per_task": "WEBFOVEA_MAX_FIND_TEXT_PER_TASK",
    "find_text_timeout_ms": "WEBFOVEA_FIND_TEXT_TIMEOUT_MS",
    "enable_read_text": "WEBFOVEA_ENABLE_READ_TEXT",
    "max_read_text_per_task": "WEBFOVEA_MAX_READ_TEXT_PER_TASK",
    "read_text_max_chars": "WEBFOVEA_READ_TEXT_MAX_CHARS",
    "enable_fill": "WEBFOVEA_ENABLE_FILL",
    "fill_timeout_ms": "WEBFOVEA_FILL_TIMEOUT_MS",
    "enable_type_newline_submit": "WEBFOVEA_ENABLE_TYPE_NEWLINE_SUBMIT",
    "log_usage": "WEBFOVEA_LOG_USAGE",
}

# Boolean fields get their own list: the positive-integer parser makes no sense for
# True/False, so the two cannot share a code path.
_BOOL_FIELDS = {"single_tab_reset", "enable_back", "enable_find_text",
                "enable_read_text", "enable_fill",
                "enable_type_newline_submit", "log_usage",
                "enable_iframe_click_fallback", "enable_provenance_rule", "enable_click_many"}

# Fields that may be 0. Here 0 is a meaningful value ("never restart" / "do not
# reopen the start page after a disconnect"), not a typo; the positive-integer
# parser would reject it and the kill switch could never be turned off.
_NONNEG_FIELDS = {"max_worker_respawns", "run_budget_s", "max_reconnect_per_task"}


@dataclass(frozen=True)
class RuntimeConfig:
    """Parameters that stay fixed for the whole run. Frozen on purpose: once a run
    has started, nobody should change them.

    The defaults below are conservative fallbacks for when config.json is missing.
    The values actually used are in config.json, and several differ from these
    defaults (max_steps, task_budget_s, max_reconnect_per_task, run_budget_s).
    """

    #: Step limit per task. The default of 40 was derived from a fixed wall-clock
    #: budget per run divided across tasks and workers, at a measured 10-25 s per step.
    #: The reference implementation uses ceil(reference_length * 2) = 30 steps; we do
    #: not follow it, because the real constraint is total time, not the reference
    #: step count (see reference_length_for()).
    #:
    #: config.json sets **80**. In a paired comparison on tasks that had exhausted
    #: the 40-step budget, all of them came back empty at 40 steps, while at 80 steps
    #: some finished with a correct answer *after* step 40, a real gain rather than
    #: noise. An earlier test of a higher limit had shown no gain, but it ran on an
    #: older version; the fixes that came later (scroll-breaker false positives,
    #: special-token stripping, repeated typing into a broken box) are what let extra
    #: steps turn into progress. **Such conclusions depend on the version; re-test
    #: after changing the code.**
    max_steps: int = 40

    #: Wall-clock budget per task (seconds). A step limit cannot catch a hang: the
    #: worst single step is a 120 s model timeout x several retries plus backoff, so
    #: a stuck task could eat the time of many others.
    #:
    #: config.json sets **1800**, raised together with max_steps=80, because at 900 s
    #: the time limit bites before the step limit: tasks were cut off at under 40
    #: steps. 1800 = 80 steps x ~20 s per step, so the *step* limit is the one that
    #: binds. Tasks that ran far past that without an answer did not succeed anyway;
    #: more time only turns a fast failure into a slow one.
    task_budget_s: int = 900

    #: How many times per task to recover from a lost browser by reopening the
    #: task's start page. Unlimited would be worse: starting over spends steps
    #: already used, and doing it repeatedly burns the whole budget.
    #: 0 = no recovery: a lost browser ends the task as FAIL_BROWSER_LOST and
    #: result.json is still written. Reopening the start page amounts to restarting
    #: the task, and in hundreds of local runs this path never triggered, so turning
    #: it off costs next to nothing. **config.json sets 0.** The default stays at 1
    #: only to match the reference behaviour when config.json is missing.
    max_reconnect_per_task: int = 1

    #: Fallback when the task file has no reference_length field. 15 is the reference
    #: implementation's default, so for the same input our result.json matches its
    #: output exactly. See reference_length_for().
    default_reference_length: int = 15

    #: Minimum "settle" time after navigation (ms); see
    #: web_controller.settle_after_navigation(). Must not be 0: some sites reach
    #: networkidle first and only draw their content with JS a moment later, and
    #: without a floor we would wait less than the old fixed 3 seconds.
    settle_floor_ms: int = 1000

    #: Maximum settle time (ms). Pages with polling, analytics beacons or WebSockets
    #: never reach networkidle; without a cap we would wait until the timeout. 8 s is
    #: the worst-case extra cost per page load.
    settle_ceil_ms: int = 8000

    #: Whether each task starts from a single tab (see web_controller.close_other_pages).
    #: **This is a kill switch, not ordinary configuration.** Closing every tab can
    #: close the browser itself; our implementation opens the new tab before closing
    #: the others, so at least one page exists at all times and that cannot happen.
    #: Still, if FAIL_BROWSER_LOST ever clusters right after the reset, set this to
    #: false to get the old behaviour back **without touching code**.
    #: The cost of turning it off is small: capture.json is already isolated per task
    #: by unbinding the RequestCollector listeners; closing tabs is a second layer that
    #: only keeps the previous task's popups and session state away from the model.
    single_tab_reset: bool = True

    #: How many times a task may be claimed by a worker.
    #: **Keep it at 1: a task that has started is never re-run from scratch.**
    #: With a value above 1 the behaviour is exactly that: a worker crashes mid-task
    #: -> the task returns to the shared pool -> another worker claims it ->
    #: `safe_create_directory` deletes the existing trajectory with `shutil.rmtree`
    #: -> the task **restarts from step 0**. That is a task-level reset, not a
    #: step-level retry, and many benchmark protocols forbid it.
    #:
    #: Cost: the task a crashed worker was holding counts as failed (recorded as
    #: abandoned in summary.json). Load balancing across the shared queue (handing
    #: tasks that have not started to idle workers) is unaffected: that is
    #: assignment, not a reset.
    max_task_attempts: int = 1

    #: How many times each worker process may be restarted. Without restarts a
    #: crashed worker is a permanently lost slot, 1/N of the throughput against a
    #: fixed time budget.
    #: **Kill switch**: 0 turns restarts off. The cap must exist, or a slot whose CDP
    #: address is gone for good loops forever through start -> init failure -> exit
    #: -> restart.
    max_worker_respawns: int = 2

    #: Retries for browser initialization (connect_over_cdp), with backoff between
    #: them. The reference implementation tried once and returned on failure, so one
    #: transient hiccup cost a whole shard. The shared queue now picks up the tasks
    #: anyway; retrying only avoids losing a slot for nothing.
    worker_init_retries: int = 3

    #: After this many consecutive tasks whose page never opened, the worker exits.
    #: This guards against a risk the shared queue *introduced*: a worker whose
    #: browser is dead fails every task it claims within seconds. Under static
    #: sharding it only ruined its own share; with a shared queue it can eat
    #: everyone's tasks. Let it exit and leave the pool to healthy workers.
    max_consecutive_page_failures: int = 3

    #: Wall-clock gate for the whole run (seconds). Past it, no new tasks are handed
    #: out; running tasks finish and the workers stop. 0 = off (`expired()` is always
    #: False when deadline is None).
    #:
    #: **config.json sets 0 (off).** The gate was meant to stop before an external
    #: time limit kills the run, but that premise turned out to be wrong:
    #:   1. Each task's `result.json` is written atomically as soon as the task ends
    #:      (temp file + os.replace). A hard kill only loses the task in progress;
    #:      every finished task is already on disk.
    #:   2. In practice, a run that was cut off by an external time limit still had
    #:      every completed task scored.
    #: So there are three outcomes: not dispatched = 0; dispatched and killed = 0;
    #: dispatched and finished = a chance to score. **Dispatching weakly dominates**,
    #: and once the gate closes the workers would sit idle anyway; keeping a buffer
    #: simply throws away the tasks at the tail.
    #: ⚠️ The one real cost: a task that is cut off adds steps and time without adding
    #: a point, which matters only if ties are broken by efficiency.
    run_budget_s: int = 12600

    #: Whether back() (browser history back) is in the action space.
    #: Without it, after following a list item into a detail page, the only way back
    #: was to redo the whole search. With it, a multi-record task went from refilling
    #: an address form twice to once, and all five back() calls took the right path.
    enable_back: bool = True

    #: back() budget per task. Without a cap it becomes a new way to idle: the model
    #: takes an action that "looks like progress" and repeats it until the steps run
    #: out (asking it to stop in the prompt did not work; it has to be enforced in the
    #: executor). 15 because back() pays off when shuttling list -> detail -> list,
    #: and multi-record tasks need about 10 of those.
    max_backs_per_task: int = 15

    #: Whether find_text() is in the action space.
    #: On a long page the model can only scroll screen by screen, which can burn a
    #: dozen steps (one task needed to reach "Peru" in a long country list and
    #: scrolled until the steps ran out).
    #: **Boundary**: find_text only locates text and scrolls it into view. It must
    #: **never** call `.inner_text()` / `.text_content()` on the Locator to read a
    #: value; the answer still has to be on screen and read by the model from the
    #: screenshot.
    enable_find_text: bool = True

    #: find_text budget per task. Without a cap it becomes a new way to idle, for the
    #: same reason as max_backs_per_task. A task rarely needs to locate more than a
    #: handful of targets; 10 leaves room and stays well below max_steps.
    max_find_text_per_task: int = 10

    #: Timeout for find_text (ms). Playwright's default is 30 s, and when the text does
    #: not exist it **waits out the full timeout before raising**, exactly the kind of
    #: time sink we had been hunting. 3 seconds is enough: the text is either in the
    #: current DOM or it is not.
    find_text_timeout_ms: int = 3000

    #: Whether read_text() is in the action space: read DOM text from the rendered page
    #: to verify a value before answering. It targets wrong answers, not unfinished
    #: tasks: several tasks failed on misread values or the wrong variant of a figure
    #: after only a few steps, i.e. the agent believed it was done. What helps there
    #: is **a check before submitting**, not more steps.
    enable_read_text: bool = True

    #: Budget per task. Tighter than find_text: every call pushes text into the context,
    #: and too much of it dilutes attention (full-page text measured 2.6k-6.3k
    #: characters, on the order of a screenshot).
    max_read_text_per_task: int = 5

    #: Maximum characters returned per call. Longer text is truncated; better to give
    #: too little than to bury the decision in unrelated text.
    read_text_max_chars: int = 600

    #: Whether fill() is in the action space.
    #: It targets **unreliable input**: in one development set, 15 of 70 tasks typed
    #: exactly the same content more than once (= the previous input did not take), and
    #: one task failed once on each of its four fields. type() carries no coordinates
    #: (`cx,cy = -1,-1`): it sends ctrl+a / Backspace / keystrokes to **whatever has
    #: focus**, which depends entirely on the previous coordinate click, and a slightly
    #: missed click **is invisible in the screenshot**. fill() targets the box directly
    #: and reports the box's final content back to the model.
    #:
    #: On by default since it passed a paired comparison. The biggest risk of a new
    #: action is not that it works badly but that the model **uses it instead of a
    #: type() that was working**, so the comparison has to include tasks that already
    #: passed with type().
    enable_fill: bool = True

    #: Timeout for each Playwright call inside fill() (ms). Same reason as
    #: find_text_timeout_ms: the 30 s default turns "element covered / not
    #: interactable" into a 30-second time sink, while the answer is known within 3 s.
    fill_timeout_ms: int = 3000

    #: Whether a trailing newline in `type(content='xxx\\n')` actually presses Enter.
    #: The action space has always promised "end with \\n to submit", but click_type
    #: typed character by character and entered `\\` and `n` into the box as text. 79
    #: of 356 type() calls (22%) had this shape, including calls in tasks that passed.
    #: Same family as `<|eot_id|>`: the prompt promised behaviour the executor never
    #: implemented.
    #:
    #: A **separate switch** from enable_fill: it changes a path that passing tasks
    #: also take, so its regression risk differs from fill()'s and it must be possible
    #: to turn it off on its own. On by default since it passed a paired comparison.
    enable_type_newline_submit: bool = True

    #: When a click lands on an iframe: judge whether it worked by the fingerprint of
    #: **the child frame that was hit**, and if the physical click had no effect,
    #: dispatch a DOM click inside that frame. See the comment above child_frame_hit in
    #: web_controller (the main-frame fingerprint is blind to changes inside an iframe,
    #: and physical clicks on an iframe at the bottom of the viewport were swallowed
    #: three times in a row).
    #: Its own switch because it touches clicking, **a path every task takes**. It only
    #: does extra work when the click hits an iframe and neither frame changed, but it
    #: must still be possible to turn it off in one place.
    enable_iframe_click_fallback: bool = True

    #: Whether the answer-provenance rule goes into the prompt (see the comment on
    #: prompts.PROVENANCE_CONSTRAINT_LINE). It changes the prompt every task reads, so
    #: it gets its own switch for A/B tests and rollback.
    #: **Off by default.** In a small trial it recovered no task, and one task went from
    #: a wrong (derived) answer to an empty one: the rule pushed it to the dashboard the
    #: task named, where it got stuck in a date field. No loss in score, but no evidence
    #: of a gain either.
    enable_provenance_rule: bool = False

    #: Whether click_many() for picture-grid CAPTCHAs is on (see the comment on
    #: prompts.CLICK_MANY_ACTION_LINE). When on, the CAPTCHA hard constraint changes
    #: from "give up immediately" to "click every square in one go, then Verify".
    enable_click_many: bool = True

    #: Maximum attempts per model call, and the total retry time (seconds, from the
    #: first call). The old fixed 3 attempts with 2/4/8 s backoff (14 s in total) let
    #: three 403 "Request not allowed" responses in a row end a task at step 0; 4 of
    #: 305 tasks died that way (1.3%). These 403s are sporadic policy refusals: about
    #: 50 others in the same period all passed on the second attempt. Waiting another
    #: minute to save a task is a much better deal than losing it.
    #: The time cap covers the case where every attempt hangs until the 120 s timeout
    #: and would burn the task's whole budget: within 240 s there is room for at most
    #: one more attempt, after which the task ends as failed.
    model_call_max_attempts: int = 6
    model_call_retry_budget_s: int = 240

    #: Whether to log the token usage of every model call to the worker log.
    #: **Off by default.** A run should only produce what relates to the answer
    #: (result.json / trajectory / capture.json), not diagnostic logs. Turn it on for
    #: local runs with an environment variable: `WEBFOVEA_LOG_USAGE=1`.
    #:
    #: Why log it at all: `response.usage` was simply discarded, so there was **no way
    #: to tell whether prompt caching took effect** other than reading the code. And
    #: the code suggests it does not: the history_n=5 image window shifts by one every
    #: step, rewriting the middle of the message list so the prefix diverges there; and
    #: the OpenAI-compatible endpoint receives no cache_control markers. At a few
    #: hundred dollars per 100-task run, that is worth confirming with data.
    log_usage: bool = False

    @classmethod
    def load(cls, config_path: str | None = None, env: dict | None = None) -> "RuntimeConfig":
        """Build the configuration with the priority env var > config.json["agent"] > default.

        A problem at any level (file missing / broken JSON / not an integer) only
        **skips that source** and never raises: a configuration that cannot be read
        gives at worst suboptimal parameters, while an exception would stop the run.
        """
        env = os.environ if env is None else env
        path = _DEFAULT_CONFIG_PATH if config_path is None else config_path

        section = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                section = json.load(f).get("agent") or {}
            if not isinstance(section, dict):
                print(f"[runtime_config] the 'agent' section of config.json is not an object, ignored: {section!r}")
                section = {}
        except FileNotFoundError:
            pass
        except Exception as e:  # JSON syntax error, encoding problem, permissions...
            print(f"[runtime_config] failed to read {path}, using all defaults: {e}")

        values = {}
        for f in fields(cls):
            raw = env.get(_ENV_KEYS[f.name], section.get(f.name))
            if f.name in _BOOL_FIELDS:
                parsed = _as_bool(raw, f.name)
            elif f.name in _NONNEG_FIELDS:
                parsed = _as_int(raw, f.name, minimum=0)
            else:
                parsed = _as_int(raw, f.name, minimum=1)
            if parsed is not None:
                values[f.name] = parsed
        return cls(**values)

    def reference_length_for(self, task: dict) -> int:
        """What to write into the `reference_length` field of `result.json`.

        The field means **the reference step count for this task**. It is a property
        of the task, not something we get to choose. The reference implementation
        takes `json_item.get("reference_length", 15)`.

        The public task files (`data/example_tasks.json`, the public Protocol III
        tasks) do not have this field at all; their fields are only
        task/task_id/task_idx/website (+answer).

        So the rule is: **echo it if the task provides one, otherwise fall back to the
        reference implementation's default.** For any input, our result.json then
        matches the reference implementation's output, and nobody has to wonder
        where the number came from.

        ⚠️ This was once hard-coded to 100, a number that came neither from the task
        nor from the reference default, and it went straight into result.json. If a
        scorer used it for an efficiency metric (e.g. predict_length /
        reference_length), we would have reported a fake baseline. **Echo the
        default rather than make up a number.**
        """
        raw = task.get("reference_length")
        # bool is a subclass of int, so True would count as 1; reject it explicitly.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return self.default_reference_length
        return int(raw) if raw > 0 else self.default_reference_length


# Per-process cache: main.py and web_controller.py both need it, but config.json should be read once.
_CACHED: "RuntimeConfig | None" = None


def get_config() -> "RuntimeConfig":
    """The configuration shared by the whole process (loaded on first call).

    Tests that need a specific source should call `RuntimeConfig.load(config_path=...,
    env=...)` directly rather than go through this cache, which is for production use.
    """
    global _CACHED
    if _CACHED is None:
        _CACHED = RuntimeConfig.load()
    return _CACHED


def _as_bool(raw, name: str):
    """Convert a config value to bool; None if it cannot be recognized (= this source
    does not apply, fall through to the next).

    Environment variables are always strings, so "false" / "0" / "no" must all be
    recognized: a kill switch whose value is not understood cannot be turned off,
    which defeats its purpose.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    print(f"[runtime_config] {name}={raw!r} is not a boolean, ignoring this source")
    return None


def _as_int(raw, name: str, minimum: int = 1):
    """Convert a config value to an integer >= minimum; None if that fails (= this
    source does not apply).

    `minimum` comes in two flavours: most parameters are positive integers (steps,
    timeouts, counts, where 0 is meaningless and most likely a typo); fields in
    `_NONNEG_FIELDS` accept 0, because there 0 legitimately means "turn this off".
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        print(f"[runtime_config] {name}={raw!r} is not an integer, ignoring this source")
        return None
    if value < minimum:
        print(f"[runtime_config] {name}={value} is below the minimum {minimum}, ignoring this source")
        return None
    return value
