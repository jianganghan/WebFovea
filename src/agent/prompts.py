"""Prompt templates: action space, hard constraints and the user-prompt skeleton.

Starts from the reference implementation. The action space the model actually sees is
assembled by normal_action_space(): UITARS_NORMAL_ACTION_SPACE is the base, and each
added action's description line and usage rule are inserted according to the
switches in config.json.

Changes from the reference implementation:
  - Base UITARS_NORMAL_ACTION_SPACE: adds hover / select / reload, removes goto, and
    adds a "Hard constraints" section: no navigating by URL, stay on the given site,
    the answer must be visible on screen before finished(), submit an empty answer
    when the data cannot be reached, how to handle human-verification checks, etc.
  - Actions inserted by switch (each one an action line plus a usage rule):
      back()        browser back                                  enable_back
      find_text()   Ctrl+F-style: scrolls to text, returns no text  enable_find_text
      read_text()   reads text from the DOM to verify a value       enable_read_text
      fill()        fills the box it points at, reads the value back enable_fill
      click_many()  clicks a picture-grid CAPTCHA in one turn       enable_click_many
      provenance    take the value from the place the task names    enable_provenance_rule (off)
  - Rules say when an action MUST be used, not "use it when you think it helps":
    rule-style triggers were adopted far more often.

⚠️ Every new action must line up in three places: the action space here, parsing in
web_controller.parse_action() and execution in web_controller.excute_action(). If one
is missing the action fails silently: the model never uses it and nothing reports an error.
"""

UITARS_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 3s and take a screenshot to check for any changes.
finished()
"""

UITARS_CALL_USR_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 3s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
"""

#: Action line and usage rule for back(). Kept separate and spliced in by switch, not
#: hard-coded, so that "with the switch off the model never learns the action exists"
#: is something a test can check. enable_back is on by default.
BACK_ACTION_LINE = (
    "back() # Go back to the PREVIOUS page you came from, exactly like the browser's "
    "back button. Use it to return to a list or results page after opening one entry, "
    "instead of redoing the whole search. It does NOT undo typing or filters on the "
    "current page.")

BACK_CONSTRAINT_LINE = (
    "- After opening an item from a list to read its details, use back() to return to "
    "that list instead of re-running the search from the start. Re-doing a search you "
    "have already done wastes most of your remaining steps.")


FIND_TEXT_ACTION_LINE = (
    "find_text(text='...') # Jump straight to words you can already name - a country, a year, "
    "an option in a long dropdown, a row label, a column header, a report title. Works like "
    "Ctrl+F and also scrolls INSIDE long lists and dropdown menus, not just the page. "
    "It only MOVES THE VIEW - you still read the value off the screenshot afterwards. If the text "
    "occurs several times, calling find_text again with the SAME text jumps to the next occurrence.")

FIND_TEXT_CONSTRAINT_LINE = (
    "- The moment you can name the exact words you are hunting for, stop scrolling and use "
    "find_text(text='...'). Scrolling a long list, table or dropdown one screen at a time is "
    "the single biggest waste of steps there is - it can eat 20 steps to reach one option "
    "that find_text reaches in one. This includes options inside a dropdown menu you have "
    "already opened.\n"
    "- Use scroll() only while you still do not know what you are looking for and are just "
    "seeing what the page offers. As soon as you can name the target, switch to find_text().")


READ_TEXT_ACTION_LINE = (
    "read_text(near='...') # Read the exact text of the element containing the words you name, "
    "straight from the page. Use it to CHECK a value you already read off the screenshot - "
    "small digits, dense tables and ranked lists are easy to misread by eye. It does not "
    "move or change anything.")

READ_TEXT_CONSTRAINT_LINE = (
    "- Before finished(), if your answer is a specific value copied off the page - a number, "
    "a date, a name, a rank - call read_text(near='...') once to check you read it correctly. "
    "Name a nearby label or the value itself. Do not skip this because you feel sure: the "
    "most common way to lose a task is confidently reporting the wrong row or a misread digit.\n"
    "- read_text may come back as garbled symbols on sites that scramble their fonts. When "
    "that happens, trust the screenshot instead - it is showing you the real value.\n"
    "- read_text does NOT move the view. After checking, make sure the answer is still "
    "visible on screen before you call finished() - the evaluator reads your last screenshot.")


FILL_ACTION_LINE = (
    "fill(start_box='<|box_start|>(x1,y1)<|box_end|>', content='...') # Put text into a text box, "
    "search box or date box. Point at the BOX itself. It focuses that box, clears whatever is "
    "already in it, types your text as real keystrokes, and then tells you what the box actually "
    "contains. End content with \"\\n\" to press Enter afterwards.")

#: The trigger is written as a **rule**, not "when you think it is needed": in targeted
#: runs, rule-style triggers were followed in 5 of 6 tasks, self-assessed ones in 2 of 6.
FILL_CONSTRAINT_LINE = (
    "- To put text into any box on the page, use fill(start_box=..., content='...'). Do NOT do it "
    "with click() followed by type(), and do NOT clear a box with hotkey(key='ctrl+a'). That "
    "sequence sends the keys to whatever happens to have focus, so if your click missed the box "
    "by a few pixels the text goes nowhere - and the screenshot looks the same either way, so you "
    "cannot tell. This is the single most common way a form task is lost.\n"
    "- Use type() only when there is no box to point at.")


#: Answer-provenance rule. Several near misses shared one pattern: **the agent reached
#: the right document but read the value from a similar table next to it.** One task
#: named "the table under Commitments and Contingencies" and the model read a different
#: table (its own thought even mentioned the right note); another asked for the
#: "On-Time Arrival Percentage" and the model computed 100 minus "Percent Flights Late",
#: which does not equal the figure the page reports. None of these were control
#: problems; they were decisions about where to read. Written as a rule for the same
#: reason as FILL_CONSTRAINT_LINE. Off in the submitted configuration.
PROVENANCE_CONSTRAINT_LINE = (
    "- When the task names WHERE the value lives - a section, a note, a table under a heading, "
    "a tab, a metric or a column name - take the value from exactly that place. Do not substitute "
    "a similar-looking table elsewhere in the same document, and do not derive the value from a "
    "different metric (100 minus a 'late' percentage is not the 'on-time' percentage the page "
    "itself reports). Only if the named place truly does not exist may you fall back to the "
    "closest one, and then say so in your answer.\n"
    "- Variants of a named document type count as that type: a 10-K405 is a 10-K. When both an "
    "original filing and an amendment (e.g. 10-K/A) exist, the task means the original unless it "
    "asks for the amendment.")

PROVENANCE_FIND_TEXT_HINT = (
    " If you are not at the named place yet, use find_text(text='...') with the heading or "
    "metric name to get there before reading anything.")


#: Picture-grid CAPTCHAs: the model lists every square to click in one reply and the
#: executor clicks them in a row. Clicking one square per turn costs one model call
#: (10-20 s) each, and the challenge times out first ("time limit exceeded"); batching
#: brings a round down to a single call. The risk is the model using it on ordinary
#: controls, so both the action line and the rule restrict it to picture grids, and the
#: executor caps it at 16 points per call and 6 calls per task.
CLICK_MANY_ACTION_LINE = (
    "click_many(points='(x1,y1);(x2,y2);(x3,y3)') # ONLY for picture-grid human-verification "
    "puzzles ('select all squares with a bus'). Clicks every listed point one after another in a "
    "single turn, then shows you the result. Up to 16 points.")

CLICK_MANY_CAPTCHA_RULE = (
    "- Human-verification challenges: if the page asks you to tick a single box saying you are "
    "human, click it once and carry on. If it shows a GRID OF PICTURES and asks you to pick every "
    "one containing something (\"choose all the bags\"), do NOT click the tiles one by one - each "
    "single click costs a whole turn and the puzzle times out. Instead look at the whole grid, then "
    "issue ONE click_many(points='...') listing the centre of every matching tile. For a static grid "
    "('select all squares with X') put the Verify button as the LAST point of that same click_many, "
    "so the whole round takes one turn. For a dynamic grid ('click verify once there are none left') "
    "select the tiles first, look at the new tiles that fade in, and click Verify only when none are "
    "left. If the grid changes or a new puzzle appears, do the same again. If after 3 rounds you are "
    "still on the puzzle, call finished(content='').")


def normal_action_space(enable_back: bool = False, enable_find_text: bool = False,
                        enable_read_text: bool = False, enable_fill: bool = False,
                        enable_provenance_rule: bool = False,
                        enable_click_many: bool = False) -> str:
    """The action space in effect. Each switch decides whether its action appears.

    ⚠️ Action space, parser and executor must line up; if one is missing the action
    **fails silently**. The only symptom is that the model never uses the new action,
    which is very hard to trace from run results. See the consistency tests in
    tests/test_back_action.py.
    """
    space = UITARS_NORMAL_ACTION_SPACE
    if enable_back:
        space = space.replace("wait() #", BACK_ACTION_LINE + "\nwait() #", 1)
        space = space.replace("- There is no developer console",
                              BACK_CONSTRAINT_LINE + "\n- There is no developer console", 1)
    if enable_find_text:
        space = space.replace("wait() #", FIND_TEXT_ACTION_LINE + "\nwait() #", 1)
        space = space.replace("- There is no developer console",
                              FIND_TEXT_CONSTRAINT_LINE + "\n- There is no developer console", 1)
    if enable_fill:
        # Placed *before* the type() line so the action space itself reads in the
        # intended order: fill when there is a box, type only when there is none.
        space = space.replace("type(content='')", FILL_ACTION_LINE + "\ntype(content='')", 1)
        space = space.replace("- There is no developer console",
                              FILL_CONSTRAINT_LINE + "\n- There is no developer console", 1)
    if enable_provenance_rule:
        # Placed next to "filters the task names must actually be set", which follows the
        # same idea: the place the task names is where the answer comes from. The
        # find_text hint is added only when find_text is on; never promise a missing action.
        rule = PROVENANCE_CONSTRAINT_LINE
        if enable_find_text:
            rule = rule.replace("and then say so in your answer.",
                                "and then say so in your answer." + PROVENANCE_FIND_TEXT_HINT, 1)
        anchor = "- When the task names filters, options, tabs or a search form"
        assert anchor in space
        space = space.replace(anchor, rule + "\n" + anchor, 1)
    if enable_click_many:
        # Action line goes before hotkey; the CAPTCHA hard constraint is replaced by the batch-click version
        space = space.replace("hotkey(key='') #", CLICK_MANY_ACTION_LINE + "\nhotkey(key='') #", 1)
        lines = space.split("\n")
        idx = [i for i, l in enumerate(lines) if l.startswith("- Human-verification challenges:")]
        assert len(idx) == 1
        lines[idx[0]] = CLICK_MANY_CAPTCHA_RULE
        space = "\n".join(lines)
    if enable_read_text:
        space = space.replace("wait() #", READ_TEXT_ACTION_LINE + "\nwait() #", 1)
        # Placed *before* "the answer must be on screen before finished()", so verify-then-finish reads in order
        space = space.replace("- Before calling finished(), check the CURRENT screenshot",
                              READ_TEXT_CONSTRAINT_LINE + "\n- Before calling finished(), check the CURRENT screenshot", 1)
    return space


UITARS_NORMAL_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hover(start_box='<|box_start|>(x1,y1)<|box_end|>') # Move the mouse onto a point WITHOUT clicking. Use this to reveal tooltips, hover-only values on charts, and menus that open on hover.
select(start_box='<|box_start|>(x1,y1)<|box_end|>', option='Vietnam') # Choose an option from a NATIVE dropdown (an HTML <select>). Point at the dropdown and name the option you want.
hotkey(key='') # ONLY these page-level keys work: Escape, Tab, Enter, Backspace, Delete, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, PageUp, PageDown, Home, End, ctrl+a, ctrl+c, ctrl+v, ctrl+x, ctrl+z, ctrl+f.
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
reload() # Reload the CURRENT page. Use this only when the page never finished loading - a spinner that never resolves, a blank area where content should be, or a message saying scripts could not run. It throws away anything you had opened, typed or filtered on this page, so do not use it to undo a mistake. At most 2 per task.
wait() #Sleep for 3s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Hard constraints
- You CANNOT navigate by URL. There is no address bar, no goto action, and no way to type a URL. Keystrokes are injected into the PAGE, not the browser UI, so browser-level shortcuts DO NOT WORK and will be rejected: ctrl+l, alt+d, F5, F6, F11, F12, ctrl+t, ctrl+w, ctrl+r, ctrl+shift+j, ctrl+shift+i. Never use them. To refresh a page that failed to load, use reload() - that is the only refresh that works.
- The ONLY way to move around this site is to operate what is on the screen: click links and buttons, type into the page's own fields, use its menus, tabs and filters. Reaching a page any other way does not count, even if the number you end up with is correct.
- There is no developer console and no devtools. Do not try to open them.
- If the same action produced no visible change twice in a row, the approach is wrong - change strategy (scroll to bring the control into view, hover it first, or target a different element) instead of repeating it.
- If the page itself never came up - a spinner that never resolves, an empty content area, or a notice that scripts could not run - that is not something clicking or scrolling can fix. Call reload() once and look at what comes back. If two reloads leave the page in the same broken state, the site is not going to work: call finished(content='').
- Human-verification challenges: if the page asks you to tick a single box saying you are human, click it once and carry on. But if it shows a GRID OF PICTURES and asks you to pick every one containing something ("choose all the bags"), stop and call finished(content='') right away. Those puzzles are timed, each click costs you a whole turn, and the timer runs out long before you can finish - the only thing you can win there is a message saying "time limit exceeded". Every step you spend on one is taken from the other tasks.
- When the task names filters, options, tabs or a search form, you must actually set them through the page's own controls. Do not look for a shortcut that skips them.
- Stay on the website you were given. Do not visit search engines, cached copies, archives, news articles or any other site - not even to find your way back. If this site blocks you or the data cannot be reached by operating the page, call finished(content='') with an empty answer.
- Some values only appear on hover - chart points, map regions, truncated labels, icons. If a number is not written on the page, hover() over the element that should carry it and read the tooltip.
- A native dropdown opens its list OUTSIDE the page, so that list never appears in your screenshot and clicking it can never work. Use select(start_box=..., option='...') for those. If you clicked something that looks like a dropdown and nothing appeared, that is what happened - switch to select(). Dropdowns whose options you CAN see drawn on the page are ordinary elements: click those normally.
- Before calling finished(), check the CURRENT screenshot: the answer must be visible on it right now, with every filter, tab, dropdown and sort the task asked for still in its selected state. Being on the right URL is NOT enough - the evaluator reads the final screenshot, not your earlier ones.
- If you already read the value but the view has since changed (a tab switched back, a popup you closed held the value, you scrolled away, a filter reset), do NOT call finished() yet. Restore that exact view first, confirm the value on screen, and only then finish.
- If you have already verified the answer on screen but cannot get back to that view after two attempts, still call finished() with the answer you verified. Submitting a verified answer from an imperfect view beats burning every remaining step and submitting nothing.
- finished(content='...') must carry the actual answer.
- Answer ONLY from what you actually saw on screen in this session. If the site blocked you or you could not reach the data, call finished(content='') with an empty answer - never fill it in from memory or general knowledge.
"""

UITARS_USR_PROMPT_NOTHOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 3s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
## User Instruction
{instruction}
"""

UITARS_USR_PROMPT_THOUGHT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""