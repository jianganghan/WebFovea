# tests

Offline unit tests: **no network, no browser, no model calls, zero tokens.**

```bash
python -m unittest discover -s tests -v      # from the repository root
```

They use only the standard library's `unittest` and the packages already in `requirements.txt`.

Most of these tests target failures that are **silent** in a real run: nothing raises, the agent just quietly does worse. An action the prompt describes but the parser never wires up, a listener that leaks into the next task, a worker crash that drops a few tasks. These are nearly impossible to reproduce reliably with a live browser, so the tests make them deterministic.

| File | What it covers | Why it is worth testing |
|---|---|---|
| `test_capture_isolation.py` | `capture.json` holds only the current task's requests: listeners unbound across tasks + single-tab reset | Cross-task contamination is very hard to reproduce in a real browser |
| `test_runtime_config.py` | Priority and fault tolerance of runtime parameters; what `reference_length` echoes | A wrong parameter can cost a whole run; a broken config must fall back to defaults and keep going |
| `test_select_action.py` | Native `<select>`: model output -> parse -> option chosen | This action fails **silently**; with no negative feedback the model retries until the steps run out |
| `test_reload_action.py` | `reload()` parsing, per-task budget, reset between tasks, timeouts not fatal; **consistency across the three places an action lives** | If the action space, parser and executor disagree, the action fails silently and only shows up as "the model never uses it" |
| `test_settle_after_navigation.py` | Post-navigation wait: floor / early exit / ceiling | Without a floor it waits less than the old fixed 3 s; without a ceiling, polling pages hang until the timeout |
| `test_task_dispatch.py` | Shared task queue: load balancing, crash compensation, poison-task cap, wall-clock gate, abandoned list; including the Manager-proxy backend | Every failure mode here is "a few tasks went missing", invisible when only a total score comes back |
| `test_boot_banner.py` | Startup banner carries the identifying information, the `BOOT OK` marker, and does not raise when a dependency is missing | A crash before startup is otherwise a complete blind spot |
| `test_special_token_strip.py` | Strips self-generated `<\|eot_id\|>` / `<\|end_of_turn\|>`; **`<\|box_start\|>` must survive** | 15 of 305 past episodes were contaminated and 14 scored zero; stripping the box tokens by mistake would break every click |
| `test_fill_action.py` | `fill()`: four-way alignment, iframe coordinate conversion, clear-then-type, both newline forms, **pointing back to click + type when no box is found** | Many tasks typed the same content repeatedly (= input silently failed); a new action must not lead the model off a path that works |
| `test_type_newline_submit.py` | A trailing newline in `type(content='x\n')` presses Enter; **switched off, behaviour is byte-identical to before** | 22% of type() calls had this shape and Enter was never pressed; passing tasks take this path too, so it needs its own switch |
| `test_iframe_click.py` | Clicks / hovers on an iframe: the **hit child frame's** fingerprint decides whether it worked, swallowed physical events get a DOM-event fallback inside that frame; hover and chart elements also compare the text hash; **plain main-frame clicks unchanged**; focus moving into an iframe is not a reaction; kill switch | The main-frame fingerprint is blind to changes inside iframes, and physical clicks on an iframe at the bottom of the viewport were swallowed three times in a row |
| `test_text_in_frames.py` | `find_text()` / `read_text()` search child frames when the main frame has no match; **no child frame touched on a main-frame match**; cross-origin elements scrolled with JS; **the same text again jumps to the next match** | Most find_text misses were content inside iframes, which `page.get_by_text` cannot see |
| `test_provenance_rule.py` | The answer-provenance prompt rule: byte-identical when off; never promises find_text when find_text is off; **off in the shipped config** | A prompt change every task reads should not ship without evidence |
| `test_model_call_retry.py` | Model-call attempts and total retry time are configurable; the old 3-attempt limit reproduces the original failure; the time cap stops "every attempt times out" | Three sporadic 403s in a row used to end a task at step 0 |
| `test_captcha_bailout.py` | Picture-grid CAPTCHAs: give up at once; **single-checkbox checks still get clicked**; switch state in the shipped config | One CAPTCHA could burn 40 steps of a shared time budget, but a blanket rule would also drop the Cloudflare checkbox |
| `test_cloudflare_checkbox.py` | A real model output for the Cloudflare checkbox lands inside the box after the coordinate fix and outside it before | Pins the coordinate mapping with the physical facts of one real frame |
| `test_click_many.py` | `click_many()`: alignment, coordinate remapping, budgets, step accounting | It exists only for CAPTCHAs and must not become a general batching shortcut |
| `test_back_action.py` / `test_find_text_action.py` / `test_read_text_action.py` | The switch, parsing, budgets and feedback of each action; find_text never reads values from the DOM; read_text's anti-scraping-font guard | Each of these was added after a concrete failure, and each can fail silently |
| `test_trailing_junk.py` | Junk after an action (e.g. `\n<br>`) no longer empties `agent_answer` | A correct answer once scored zero at the extraction step |
| `test_score.py` | Verdict tiers of the local scorer `eval/score.py`: all date formats the FAQ calls equivalent, no over-matching, ambiguous dates, end-to-end wiring | Each misjudged answer becomes a phantom "wrong answer" in failure analysis |

`fakes.py` holds Playwright stand-ins that implement only the methods the code under test uses. They turn timing problems such as "listener not unbound" or "pages closed down to zero" into assertable state.

Some Chinese strings remain in the test data on purpose: the WebRetriever tasks are mostly Chinese, so the scorer must accept Chinese date formats and ordinary Chinese text must never be flagged as obfuscated.

The JavaScript halves of the iframe click fallback and of `fill()` (`elementFromPoint` plus sibling / ancestor lookup, native date inputs) need a real browser; they were verified separately on fixed local `file://` pages.

Not tested here: rendering and bot protection on real sites, and whether the model chooses to use a given action. The first needs a real browser (`scripts/run_local.sh` on a single task); the second is a matter for the prompt.
