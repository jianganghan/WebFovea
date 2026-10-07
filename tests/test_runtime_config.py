"""Priority and fault tolerance of runtime parameters.

These parameters decide how long and how many steps a task runs and what result.json reports;
getting them wrong can cost a whole run. So the tests fall into two groups:
  1. **Semantics**: what reference_length should echo, and what it must never make up;
  2. **Fault tolerance**: a missing / broken config file or an invalid value must fall back to the
     default and keep going, never raise. An unreadable config gives at worst suboptimal parameters;
     an exception would stop the run from starting.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agent"))

from runtime_config import RuntimeConfig  # noqa: E402

REPO_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.json")


def write_config(tmpdir, payload):
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


class TestPrecedence(unittest.TestCase):
    """Env var > config.json["agent"] > default in code."""

    def test_defaults_when_nothing_configured(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertEqual(cfg.max_steps, 40)
        self.assertEqual(cfg.task_budget_s, 900)
        self.assertEqual(cfg.max_reconnect_per_task, 1)
        self.assertEqual(cfg.default_reference_length, 15)

    def test_config_file_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_config(d, {"api_base": "x", "agent": {"max_steps": 25}})
            cfg = RuntimeConfig.load(config_path=path, env={})
            self.assertEqual(cfg.max_steps, 25)
            self.assertEqual(cfg.task_budget_s, 900)   # fields not configured keep their default

    def test_env_overrides_config_file(self):
        """A local diagnostic run that raises the step limit should not have to edit the shipped
        config.json (and risk forgetting to change it back)."""
        with tempfile.TemporaryDirectory() as d:
            path = write_config(d, {"agent": {"max_steps": 25}})
            cfg = RuntimeConfig.load(config_path=path, env={"WEBFOVEA_MAX_STEPS": "100"})
            self.assertEqual(cfg.max_steps, 100)

    def test_repo_config_is_parseable(self):
        """The real config.json in the repo must parse to the expected values: a hand-edited mistake
        (an extra comma, a value written as a string) should be caught here before it ships."""
        cfg = RuntimeConfig.load(config_path=REPO_CONFIG, env={})
        self.assertEqual(cfg.max_steps, 80)
        self.assertEqual(cfg.task_budget_s, 1800)
        self.assertIs(cfg.single_tab_reset, True)
        self.assertEqual(cfg.max_reconnect_per_task, 0)   # do not reopen the start page after a disconnect; see TestReconnectSwitch


class TestSingleTabResetSwitch(unittest.TestCase):
    """`single_tab_reset` is a kill switch: closing every tab can close the browser itself, and if that
    ever happens it must be possible to **get the old behaviour back by changing the configuration**,
    without touching code. A switch that cannot be turned off is useless, so value parsing must be lenient.
    """

    def test_defaults_to_on(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent", env={})
        self.assertIs(cfg.single_tab_reset, True)

    def test_can_be_turned_off_from_config_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_config(d, {"agent": {"single_tab_reset": False}})
            self.assertIs(RuntimeConfig.load(config_path=path, env={}).single_tab_reset, False)

    def test_accepts_the_string_spellings_an_env_var_can_carry(self):
        """Environment variables are always strings. Turning it off on short notice goes through this path."""
        for text, expected in [("false", False), ("0", False), ("no", False), ("off", False),
                               ("true", True), ("1", True), ("yes", True), ("ON", True)]:
            cfg = RuntimeConfig.load(config_path="/nonexistent",
                                     env={"WEBFOVEA_SINGLE_TAB_RESET": text})
            self.assertIs(cfg.single_tab_reset, expected, f"{text!r} was not parsed correctly")

    def test_unrecognised_value_falls_back_to_on_rather_than_off(self):
        """Unrecognized values fall back to the default (on) rather than quietly switching off: one typo must not silently change behaviour."""
        cfg = RuntimeConfig.load(config_path="/nonexistent",
                                 env={"WEBFOVEA_SINGLE_TAB_RESET": "maybe"})
        self.assertIs(cfg.single_tab_reset, True)

    def test_a_positive_int_field_is_not_parsed_as_bool(self):
        """Boolean and positive-integer fields take different parsing paths; keep them apart."""
        cfg = RuntimeConfig.load(config_path="/nonexistent", env={"WEBFOVEA_MAX_STEPS": "true"})
        self.assertEqual(cfg.max_steps, 40)


class TestFaultTolerance(unittest.TestCase):
    """A problem at any level only skips that source and never raises."""

    def test_missing_agent_section_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            # If config.json is replaced (keeping only the api_* fields), we must still run normally
            path = write_config(d, {"api_base": "x", "api_key": "y"})
            self.assertEqual(RuntimeConfig.load(config_path=path, env={}).max_steps, 40)

    def test_malformed_json_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{ this is not json")
            self.assertEqual(RuntimeConfig.load(config_path=path, env={}).max_steps, 40)

    def test_non_numeric_and_non_positive_values_are_ignored(self):
        for bad in ("abc", "", "0", "-5", None):
            cfg = RuntimeConfig.load(config_path="/nonexistent", env={"WEBFOVEA_MAX_STEPS": bad}
                                     if bad is not None else {})
            self.assertEqual(cfg.max_steps, 40, f"bad value {bad!r} must not take effect")

    def test_agent_section_of_wrong_type_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_config(d, {"agent": "40"})
            self.assertEqual(RuntimeConfig.load(config_path=path, env={}).max_steps, 40)


class TestReferenceLength(unittest.TestCase):
    """What reference_length in result.json should contain.

    It means "the reference step count for this task", a property of the task. The reference implementation
    takes `json_item.get("reference_length", 15)`; our rule is **echo it if the task provides one,
    otherwise fall back to the same default**, so for any input the output matches the reference implementation.
    """

    def setUp(self):
        self.cfg = RuntimeConfig.load(config_path="/nonexistent", env={})

    def test_echoes_the_value_the_task_file_gives(self):
        self.assertEqual(self.cfg.reference_length_for({"reference_length": 22}), 22)

    def test_falls_back_to_reference_default_when_absent(self):
        """Regression: this was once hard-coded to 100, a number that came neither from the task nor from the
        reference default, and it went straight into result.json. If a scorer used it for an efficiency metric
        (predict_length / reference_length), we would have reported a fake baseline.

        The public task files (example_tasks.json and the public Protocol III tasks) have no such field,
        so this fallback is the path **every task** takes in local runs.
        """
        task = {"task_idx": 0, "task_id": "abc", "task": "…", "website": "https://x"}
        got = self.cfg.reference_length_for(task)
        self.assertEqual(got, 15)
        self.assertNotEqual(got, 100, "the made-up 100 is back")

    def test_rejects_junk_values(self):
        for junk in (True, False, None, "15", [], {}, 0, -3):
            self.assertEqual(self.cfg.reference_length_for({"reference_length": junk}), 15,
                             f"{junk!r} must not count as a valid reference step count")

    def test_default_is_configurable(self):
        """If the convention ever changes (or should follow total_steps), change the configuration, not the code."""
        with tempfile.TemporaryDirectory() as d:
            path = write_config(d, {"agent": {"default_reference_length": 30}})
            cfg = RuntimeConfig.load(config_path=path, env={})
            self.assertEqual(cfg.reference_length_for({}), 30)


class TestDispatchKnobs(unittest.TestCase):
    """The four dispatch fault-tolerance parameters. What they have in common: when they go wrong, a few
    tasks quietly go missing, which a run that only reports a total score does not reveal."""

    def test_defaults(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertEqual(cfg.max_task_attempts, 1)   # a task that has started is never re-run
        self.assertEqual(cfg.max_worker_respawns, 2)
        self.assertEqual(cfg.worker_init_retries, 3)
        self.assertEqual(cfg.max_consecutive_page_failures, 3)

    def test_respawn_can_be_switched_off_with_zero(self):
        """0 is a legitimate value for the kill switch ("never restart"), not a typo.

        A positive-integer parser would ignore 0 as invalid, and the switch could then **never be turned
        off** while looking configurable.
        """
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_MAX_WORKER_RESPAWNS": "0"})
        self.assertEqual(cfg.max_worker_respawns, 0)
        with tempfile.TemporaryDirectory() as d:
            path = write_config(d, {"agent": {"max_worker_respawns": 0}})
            self.assertEqual(RuntimeConfig.load(config_path=path, env={}).max_worker_respawns, 0)

    def test_zero_is_still_rejected_for_the_others(self):
        """For other fields 0 is meaningless (claiming a task 0 times = running nothing) and must fall back to the default."""
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_MAX_TASK_ATTEMPTS": "0",
                                      "WEBFOVEA_WORKER_INIT_RETRIES": "0",
                                      "WEBFOVEA_MAX_CONSECUTIVE_PAGE_FAILURES": "-1"})
        self.assertEqual(cfg.max_task_attempts, 1)
        self.assertEqual(cfg.worker_init_retries, 3)
        self.assertEqual(cfg.max_consecutive_page_failures, 3)

    def test_task_level_retry_is_off_by_default(self):
        """A task that has started must never be retried or reset as a whole.
        The default must be 1. A value above 1 still parses, but it is a configuration that re-runs tasks;
        this test guards **safe by default**, so nobody forgets to change it back."""
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertEqual(cfg.max_task_attempts, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestUsageLoggingIsOffByDefault(unittest.TestCase):
    """This switch has exactly one correct default: off.

    A run should only produce what relates to the answer (result.json / trajectory / capture.json),
    not diagnostic logs. Turn it on temporarily for local runs with an environment variable.
    This test guards against someone writing it into config.json one day and shipping it.
    """

    def test_default_is_off(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json", env={})
        self.assertFalse(cfg.log_usage)

    def test_the_shipped_config_does_not_turn_it_on(self):
        import json as _json
        raw = _json.load(open(REPO_CONFIG, encoding="utf-8")).get("agent", {})
        self.assertFalse(raw.get("log_usage", False),
                         "the shipped config.json must not turn on log_usage; a run should not produce diagnostic logs")

    def test_env_var_can_turn_it_on_locally(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_LOG_USAGE": "1"})
        self.assertTrue(cfg.log_usage)


class TestReconnectSwitch(unittest.TestCase):
    """It must be possible to turn off "reopen the start page and continue" after a disconnect.

    Reopening the start page amounts to restarting the task, and in hundreds of local runs it never
    triggered. So 0 must be a valid value (a positive-integer parser would ignore it as a typo and
    quietly fall back to 1, exactly the silent failure to prevent), and the shipped config.json sets 0.
    """

    def test_zero_is_accepted_from_config_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_config(d, {"agent": {"max_reconnect_per_task": 0}})
            cfg = RuntimeConfig.load(config_path=path, env={})
            self.assertEqual(cfg.max_reconnect_per_task, 0)

    def test_zero_is_accepted_from_env(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_MAX_RECONNECT_PER_TASK": "0"})
        self.assertEqual(cfg.max_reconnect_per_task, 0)

    def test_negative_is_still_rejected(self):
        cfg = RuntimeConfig.load(config_path="/nonexistent/config.json",
                                 env={"WEBFOVEA_MAX_RECONNECT_PER_TASK": "-1"})
        self.assertEqual(cfg.max_reconnect_per_task, 1)   # an invalid value falls back to the default

    def test_repo_config_disables_reconnect(self):
        cfg = RuntimeConfig.load(config_path=REPO_CONFIG, env={})
        self.assertEqual(cfg.max_reconnect_per_task, 0)
