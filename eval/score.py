#!/usr/bin/env python3
"""Score an agent run against WebRetriever Protocol III reference answers.

    python eval/score.py --runs runs/my_run --gold path/to/protocol3.json [--fuzzy]

The gold file is protocol3.json from the WebRetriever dataset on Hugging Face
(https://huggingface.co/datasets/Mininglamp-2718/WebRetriever, gated, CC BY 4.0),
or any task list with the same fields plus an ``answer`` field.

Four tiers, tried in order; a later tier runs only if the earlier ones fail:
  exact  : equal after normalisation (or one contains the other)
  date   : the reference answer is a date -> compare (year, month, day)
  numeric: compare the numbers in order, within a relative tolerance (default 0.5%)
  fuzzy  : optional LLM judgement of semantic equivalence (--fuzzy, needs ANTHROPIC_API_KEY)

⚠️ This is an approximation of the official metric, not a reproduction. The
challenge used an LLM judge plus manual review and is more lenient than the rule
tiers here, so local numbers are not comparable with leaderboard numbers. Also
note that some reference answers in the public set have gone stale because the
live sites changed.

Only ``agent_answer`` in result.json is read, because that is the field the
official scorer reads. Writes <runs>/eval_answers.json.
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

#: Verdicts that count as a pass. Single definition point, so any other script
#: that filters on "passed" uses the same set.
PASS_VERDICTS = ("exact", "date", "numeric", "fuzzy")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\s《》「」『』\"'，。、：:；;！!？?（）()\[\]]+", "", s)
    return s.lower()


# ── date tier ──────────────────────────────────────────────────────────────
# The challenge FAQ states that for a reference answer of `2024年3月15日`, both
# `2024/03/15` and `March 15, 2024` are correct. Without this tier, most such
# formats fail the exact tier and are mangled by the numeric tier
# (`2024-03-15` -> [2024, -3, -15]), so correct date answers would be counted
# as wrong.
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december"]
_MONTHS = {}
for _i, _n in enumerate(_MONTH_NAMES, 1):
    _MONTHS[_n] = _i
    _MONTHS[_n[:3]] = _i
_MONTHS["sept"] = 9

_DATE_PATTERNS = [
    # 2024年3月15日 / 2024 年 03 月 15 号
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]"), "ymd"),
    # 2024/03/15  2024-03-15  2024.03.15 -- year first, unambiguous
    (re.compile(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})(?!\d)"), "ymd"),
    # 03/15/2024  15-03-2024 -- year last, day/month order ambiguous; see parse_dates
    (re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})"), "ambiguous"),
    # March 15, 2024 / Mar 15 2024 / March 15th, 2024
    (re.compile(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I), "mdy_name"),
    # 15 March 2024 / 15th Mar, 2024
    (re.compile(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", re.I), "dmy_name"),
]


def _mk(y, m, d):
    """Keep only calendar-plausible combinations so junk does not enter the candidate set."""
    if 1000 <= y <= 9999 and 1 <= m <= 12 and 1 <= d <= 31:
        return (y, m, d)
    return None


def parse_dates(s: str) -> set:
    """Parse every recognisable date in ``s`` into {(year, month, day), ...}.

    A set rather than a single value because `05/03/2024` is genuinely ambiguous
    (3 May or 5 March); both readings go in, and matching either one counts.
    Leniency in this direction is deliberate: a local scorer that systematically
    under-counts produces phantom "wrong answer" cases in failure analysis, which
    is worse than being one notch too lenient on rare ambiguous dates.
    """
    out = set()
    text = str(s)
    for pat, kind in _DATE_PATTERNS:
        for g in pat.findall(text):
            try:
                if kind == "ymd":
                    v = _mk(int(g[0]), int(g[1]), int(g[2]))
                    if v:
                        out.add(v)
                elif kind == "ambiguous":
                    a, b, y = int(g[0]), int(g[1]), int(g[2])
                    for v in (_mk(y, a, b), _mk(y, b, a)):
                        if v:
                            out.add(v)
                elif kind == "mdy_name":
                    m = _MONTHS.get(g[0].lower())
                    v = _mk(int(g[2]), m, int(g[1])) if m else None
                    if v:
                        out.add(v)
                elif kind == "dmy_name":
                    m = _MONTHS.get(g[1].lower())
                    v = _mk(int(g[2]), m, int(g[0])) if m else None
                    if v:
                        out.add(v)
            except (ValueError, TypeError):
                continue
    return out


def date_match(pred: str, gold: str) -> bool:
    """Compare (year, month, day) instead of strings, but only when the reference
    answer itself parses as a date; otherwise stray numbers in an answer such as
    "Near and Far, Geek Rating 7.226" could be assembled into a fake date.
    """
    g = parse_dates(gold)
    if not g:
        return False
    return bool(g & parse_dates(pred))


def extract_numbers(s: str):
    # Known quirk: `-?` reads a hyphen as a minus sign ("2024-03-15" -> [2024, -3, -15]).
    # Dates are handled by the date tier above. This is left unchanged so verdicts
    # stay comparable with every earlier run that was scored with it.
    return [float(x.replace(",", "")) for x in re.findall(r"-?[\d,]+\.?\d*", str(s)) if x.strip(",")]


def numeric_match(pred: str, gold: str, rtol=0.005) -> bool:
    """Every number in the reference must appear, in order, in the prediction (within rtol)."""
    gn, pn = extract_numbers(gold), extract_numbers(pred)
    if not gn or len(pn) < len(gn):
        return False
    i = 0
    for g in gn:
        while i < len(pn):
            p = pn[i]; i += 1
            if abs(p - g) <= max(abs(g) * rtol, 1e-9):
                break
        else:
            return False
    return True


def fuzzy_match(pred: str, gold: str, task: str, model: str) -> bool:
    from openai import OpenAI
    client = OpenAI(base_url="https://api.anthropic.com/v1", api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.chat.completions.create(
        model=model, max_tokens=8,
        messages=[{"role": "user", "content":
            f"Task: {task}\nReference answer: {gold}\nAgent answer: {pred}\n"
            "Is the agent answer factually equivalent to the reference answer? Reply YES or NO only."}])
    return "YES" in r.choices[0].message.content.upper()


def load_pred(task_dir: Path):
    """Read ``agent_answer`` from result.json, the only field the official scorer reads.

    Do not fall back to the last thought or the last action: that would turn
    "submitted no answer" into "submitted a wrong answer" and distort the analysis.
    """
    rj = task_dir / "result.json"
    if not rj.exists():
        return None
    d = json.loads(rj.read_text(encoding="utf-8"))
    return d.get("agent_answer") or None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="output directory of a run")
    ap.add_argument("--gold", required=True,
                    help="protocol3.json, or any task list whose items carry an 'answer' field")
    ap.add_argument("--fuzzy", action="store_true", help="LLM judge for answers the rule tiers reject")
    ap.add_argument("--judge-model", default="claude-sonnet-4-5")
    ap.add_argument("--rtol", type=float, default=0.005)
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        sys.exit(f"gold file not found: {gold_path}")
    gold = {t["task_id"]: t for t in json.loads(gold_path.read_text(encoding="utf-8"))}
    rows, n_ok = [], 0
    run_dir = Path(args.runs)
    for td in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name not in ("logs", "locks")):
        # Task directories are named <task_idx>_<task_id>.
        tid = td.name.split("_", 1)[-1]
        if tid not in gold:
            continue
        t = gold[tid]
        pred = load_pred(td)
        verdict = "no_answer"
        if pred and pred.strip():
            np_, ng = normalize(pred), normalize(t["answer"])
            gold_is_number = bool(re.fullmatch(r"-?[\d,]+\.?\d*%?", t["answer"].strip()))
            # Purely numeric answers skip substring matching ('34' would match '1234').
            if ng and not gold_is_number and (ng in np_ or np_ in ng):
                verdict = "exact"
            elif gold_is_number and np_ == ng:
                verdict = "exact"
            elif date_match(pred, t["answer"]):
                verdict = "date"
            elif numeric_match(pred, t["answer"], args.rtol):
                verdict = "numeric"
            elif args.fuzzy and fuzzy_match(pred, t["answer"], t["task"], args.judge_model):
                verdict = "fuzzy"
            else:
                verdict = "wrong"
        ok = verdict in PASS_VERDICTS
        n_ok += ok
        rows.append({"task_id": tid, "website": t["website"], "verdict": verdict,
                     "gold": t["answer"][:80], "pred": (pred or "")[:80]})
        print(f"[{'O' if ok else 'X'}] {verdict:9s} {t['website'][:40]:42s} "
              f"gold={t['answer'][:40]!r} pred={(pred or '')[:40]!r}")

    total = len(rows)
    print(f"\n== score: {n_ok}/{total} = {n_ok/total*100:.1f}%" if total else "no tasks found")
    (run_dir / "eval_answers.json").write_text(json.dumps(
        {"score": n_ok, "total": total, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
