"""Verdict rules of the local scorer (eval/score.py). Offline, zero tokens.

The scorer produces no results of its own, but every failure analysis rests on
it: each answer it misjudges becomes a phantom "wrong answer" that someone will
try to fix in the agent.

The date tier exists because the challenge FAQ says `2024年3月15日`,
`2024/03/15` and `March 15, 2024` are all correct for the same reference answer.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from score import date_match, parse_dates  # noqa: E402

TOOL = EVAL_DIR / "score.py"
GOLD = "2024年3月15日"


class TestFaqDateExamples(unittest.TestCase):
    def test_every_format_the_faq_calls_equivalent(self):
        for pred in ["2024/03/15", "March 15, 2024", "2024-03-15",
                     "15 March 2024", "2024.03.15", "2024年3月15日"]:
            with self.subTest(pred=pred):
                self.assertTrue(date_match(pred, GOLD), f"{pred!r} should match the reference")

    def test_common_variants_beyond_the_faq_list(self):
        for pred in ["Mar 15, 2024", "15th March 2024", "March 15th, 2024",
                     "2024年03月15日", "03/15/2024"]:
            with self.subTest(pred=pred):
                self.assertTrue(date_match(pred, GOLD))

    def test_date_embedded_in_a_sentence_still_matches(self):
        """Agents rarely return a bare date."""
        self.assertTrue(date_match("The notice was published on March 15, 2024.", GOLD))


class TestDoesNotOverMatch(unittest.TestCase):
    """Leniency applies to dates only; it must not let other answers through."""

    def test_a_different_date_is_still_wrong(self):
        self.assertFalse(date_match("2024年3月16日", GOLD))
        self.assertFalse(date_match("March 15, 2023", GOLD))

    def test_tier_stays_off_when_the_reference_is_not_a_date(self):
        self.assertFalse(date_match("2024/03/15", "Near and Far, Geek Rating 7.226"))
        self.assertFalse(date_match("anything", "27.1"))

    def test_impossible_calendar_values_are_dropped(self):
        self.assertEqual(parse_dates("2024/13/45"), set())
        self.assertEqual(parse_dates("0024/03/15"), set())

    def test_no_answer_matches_nothing(self):
        self.assertFalse(date_match("", GOLD))


class TestAmbiguousOrdering(unittest.TestCase):
    """`05/03/2024` could be 3 May or 5 March; both readings are accepted."""

    def test_both_readings_are_accepted_when_genuinely_ambiguous(self):
        self.assertEqual(parse_dates("05/03/2024"), {(2024, 5, 3), (2024, 3, 5)})

    def test_no_ambiguity_when_one_component_exceeds_twelve(self):
        self.assertEqual(parse_dates("15/03/2024"), {(2024, 3, 15)})

    def test_four_digit_year_first_is_never_ambiguous(self):
        self.assertEqual(parse_dates("2024-05-03"), {(2024, 5, 3)})


class TestEndToEndWiring(unittest.TestCase):
    """date_match must actually be wired into the verdict chain and counted as a pass.

    This guards against the silent failure "function correct, never called":
    scores keep coming out, they just never go up.
    """

    def run_scorer(self, gold_answer, agent_answer):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gold_file = d / "gold.json"
            gold_file.write_text(json.dumps([{
                "task_id": "abc123", "task_idx": 0,
                "website": "https://example.com",
                "task": "ask for a date", "answer": gold_answer,
            }], ensure_ascii=False), encoding="utf-8")

            runs = d / "runs"
            task_dir = runs / "0_abc123"
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(
                {"agent_answer": agent_answer}, ensure_ascii=False), encoding="utf-8")

            subprocess.run([sys.executable, str(TOOL), "--runs", str(runs),
                            "--gold", str(gold_file)],
                           check=True, capture_output=True)
            return json.loads((runs / "eval_answers.json").read_text(encoding="utf-8"))

    def test_date_verdict_is_reported_and_counted_as_a_pass(self):
        out = self.run_scorer(GOLD, "March 15, 2024")
        self.assertEqual(out["rows"][0]["verdict"], "date")
        self.assertEqual(out["score"], 1, "date tier not counted as a pass")

    def test_a_wrong_date_still_scores_zero(self):
        out = self.run_scorer(GOLD, "March 16, 2024")
        self.assertEqual(out["score"], 0)

    def test_existing_tiers_still_work(self):
        self.assertEqual(self.run_scorer("27.1", "Shipping costs 27.1 GBP")["rows"][0]["verdict"], "numeric")
        self.assertEqual(self.run_scorer("Near and Far", "The answer is Near and Far")["rows"][0]["verdict"], "exact")
        self.assertEqual(self.run_scorer("27.1", "")["rows"][0]["verdict"], "no_answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
