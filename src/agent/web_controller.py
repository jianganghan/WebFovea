"""Browser layer: connect, open pages, execute actions, record network requests.

Starts from the reference implementation; init_playwright_context() and its
authentication-header logic are kept as they were. Changes, in call order:

  Open      open_page()
     - Single-tab reset, close_other_pages(): each task keeps only its own tab. The
       new tab is created before the old ones are closed, so at least one page
       exists at all times (closing the last page also closes a CDP-connected browser).
     - First-paint wait, settle_after_navigation(): floor -> early exit -> ceiling,
       instead of a blind fixed 3-second wait.
  Parse     parse_action() / parse_action_type()
     Adds hover / select / fill / reload / back / find_text / read_text / click_many.
     goto is still parsed, but only so the executor can reject it.
  Execute   excute_action(), the same sequence for every action:
     fingerprint before -> dispatch by action name -> handle newly opened tabs ->
     wait for rendering -> compare fingerprints to detect a no-op
     - Feedback channel LAST_ACTION_NOTE: when an action is rejected, fails or has no
       effect, a sentence is recorded and the main loop passes it to the model.
       Without it a silently failed action is invisible to the model, which then
       repeats it until the step budget runs out.
     - No-op detection, page_fingerprint(): compares a DOM fingerprint (url / title /
       node count / text length / scroll offsets / focus), which ignores clocks and
       carousels; when nothing changed, the model is told plainly not to repeat itself.
     - iframe fallback, child_frame_hit() / js_click_in_frame(): when a click lands in
       an iframe, the child frame's fingerprint decides whether it worked; if the
       physical click was swallowed, DOM events are dispatched inside the child frame
       (except in CAPTCHA frames).
     - Hard blocks: browser-level shortcuts (ctrl+l / F5 ...) and goto are always
       rejected, with the reason.
     - Per-task budgets (rejected with a reason once exceeded): reload <= 2,
       back <= 15, find_text <= 10, read_text <= 5, click_many <= 6. Reset at the
       start of each task by reset_stall_counters().
  Record    RequestCollector
     attach() is idempotent and clear() unbinds before emptying, so capture.json
     holds only the current task's requests.

Boundaries: the agent reaches answers only by operating the page. find_text only
scrolls and never reads a value; read_text reads DOM text from the page as currently
rendered; click_many counts one step per click. goto, search engines and hand-built
API URLs are never used, and the executor rejects them outright.
"""
import json
import uuid
from playwright.sync_api import sync_playwright
import random
import platform
import base64
import json
import re
import os
import time
import unicodedata
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs

from runtime_config import get_config

# ========== Remote browser authentication ==========
# Hosted browsers may require an access token, passed as ?access_token=... in the CDP
# URL and sent as the X-Access-Token header. A local Chrome needs none of this.
# Global sandbox reference
_sandbox_instance = None

def connect_existing_sandbox(cdp_url, access_token):
    """Register the access token of a remote browser and set the global headers."""
    global _sandbox_instance, _cdp_headers
    _cdp_headers = {"X-Access-Token": access_token}
    print(f"✅ Using remote browser, CDP: {cdp_url[:60]}...")
    return cdp_url

# Internal headers cache
_cdp_headers = {}

def get_cdp_headers():
    """Get headers needed for CDP connection headers"""
    if _cdp_headers:
        return _cdp_headers
    if _sandbox_instance:
        return {"X-Access-Token": str(_sandbox_instance._envd_access_token)}
    return {}
# ========== End remote browser authentication ==========

browser_window_size_width = 1280
browser_window_size_height = 720

def extract_between_keywords(text, start_key, end_key):
    pattern = re.compile(f'{re.escape(start_key)}(.*?){re.escape(end_key)}', re.DOTALL)
    matches = pattern.findall(text)
    return [match.strip() for match in matches]

def extract_box_coordinate(response, box_name="start_box"):
    """Extract (x, y) coordinate from both formats:
    - New: start_box='<|box_start|>(x,y)<|box_end|>'
    - Old: start_box='(x,y)'
    """
    # New format with <|box_start|>...<|box_end|>
    m = re.search(rf"{box_name}='<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>'", response)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Old format
    m = re.search(rf"{box_name}='\(\s*(\d+)\s*,\s*(\d+)\s*\)'", response)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

# Initialize Playwright
def init_playwright_context(url):
    p = sync_playwright().start()
    try:
        headers = get_cdp_headers()
        # Remote browser: take access_token from the URL's query string
        if not headers:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            token = qs.get('access_token', [None])[0]
            if token:
                connect_existing_sandbox(url, token)
                headers = get_cdp_headers()
                print(f"🔑 Found an access token in the CDP URL")
        browser = p.chromium.connect_over_cdp(url, **({"headers": headers} if headers else {}))
        # Get existing context or create new
        context = browser.contexts[0] if browser.contexts else browser.new_context()
    except Exception as e:
        print("init error", e)
        return None, None, None

    return p, browser, context

# Open webpage
def settle_after_navigation(page, floor_ms=None, ceil_ms=None):
    """Wait for the page to settle after navigation. Returns the milliseconds waited.

    **Replaces the old fixed `wait_for_timeout(3000)`**, which was a blind bet that
    3 seconds would do:

        page.goto(url, wait_until="domcontentloaded")   # DOM skeleton ready
        page.wait_for_timeout(3000)                     # then wait 3 s with eyes closed

    `domcontentloaded` only means the DOM tree exists, **not that the content is
    there**. Many target sites are SPAs and data portals whose pipeline is "DOM
    skeleton + spinner -> download and run the JS bundle -> data XHR -> render once
    the data arrives". On fast sites the 3 seconds are wasted; on slow ones they are
    not enough, so the step-0 screenshot shows a spinner and the model starts working
    on a page that has not formed yet. In one development set, 8 of 70 tasks kept
    complaining that the page had not loaded, and 6 of them ended without an answer.

    Three stages instead, **floor -> early exit -> ceiling**, and all three are needed:

      floor      covers pages that reach networkidle first and only draw content with
                 JS a moment later. Without it this change would wait *less* than the
                 old fixed 3 seconds; that is the only real risk in this function.
      early exit fast sites reach networkidle in about 0.7 s; no need to sit out 3 s.
      ceiling    pages with polling / analytics beacons / WebSockets *never* reach
                 "no connections for 500 ms"; without a cap we would wait until the
                 timeout. With it, the worst case is a fixed cost.

    Not settling is not an error: a `wait_for_load_state` timeout just means this
    page never goes quiet. Carry on and let the model judge from the screenshot (it
    still has wait() and reload()).
    """
    cfg = get_config()
    floor_ms = cfg.settle_floor_ms if floor_ms is None else floor_ms
    ceil_ms = cfg.settle_ceil_ms if ceil_ms is None else ceil_ms

    started = time.time()
    page.wait_for_timeout(floor_ms)
    for state in ("load", "networkidle"):
        remaining = ceil_ms - (time.time() - started) * 1000
        if remaining <= 0:
            break
        try:
            page.wait_for_load_state(state, timeout=remaining)
        except Exception:
            break          # this page never goes quiet (long-lived connections / polling / beacons); stop waiting
    return int((time.time() - started) * 1000)


def close_other_pages(context, keep):
    """Keep only the `keep` tab and close all others. Returns the number actually closed.

    A "single-tab reset" at the start of every task solves two problems:
      1. **capture.json contamination across tasks**: pages left over from the
         previous task keep sending requests in the background (polling, beacons,
         timed refreshes), and those would be recorded in this task's capture.json,
         where requests to another site make no sense. Unbinding listeners in
         RequestCollector.clear() is the first layer; closing the pages themselves is
         the second. A page that no longer exists leaves no trace at all.
      2. **State interference**: popups, login state or half-filled forms from the
         previous task sit in neighbouring tabs, and the model gets led astray if it
         switches to one.

    ⚠️ Create the new page before calling this. When a context drops to 0 pages, a
    CDP-connected browser is closed along with it, which is why the reference
    implementation kept one old page (`pages[:-1]`) rather than closing them all.
    Creating first and closing second removes that window, so everything can go.

    Failing to close a single page is not a failure: at worst one extra tab stays
    open, which must not bring down the whole task.
    """
    closed = 0
    for page in list(context.pages):
        if page is keep:
            continue
        try:
            try:
                page.evaluate("() => { window.stop(); }")  # stop in-flight requests first
            except Exception:
                pass
            page.close(run_before_unload=False)  # skip onbeforeunload confirmation dialogs
            closed += 1
        except Exception as e:
            print(f"Failed to close leftover page: {e}")
    return closed


def open_page(p, browser, context, cdp_url, target_url, max_retries = 3):
    """Open the start page of a task: new tab (reconnecting CDP if the connection is
    broken) -> close other tabs -> size the viewport to the browser window -> add
    no-cache headers -> goto the start URL -> wait for first paint. Returns the new page.

    ⚠️ Called only at the start of a task (and for disconnect recovery when
    max_reconnect_per_task > 0; the shipped config sets 0). No model action can
    trigger it: "go back to the start page" is not in the action space.
    """
    # --- Fix code ---
    # Strip whitespace
    if target_url:
        target_url = target_url.strip()
    # Auto-complete https
    if target_url and not target_url.startswith("http"):
        target_url = "https://" + target_url
    # ----------------

    # print(f"🌐 Opening page: {target_url}", context)
    print(f"🌐 Opening page: {target_url}")
    page = None
    
    retry_count = 0
    while retry_count < max_retries:
        # Old pages are cleaned up only after the new page exists (close_other_pages
        # below). The order matters: at 0 pages, CDP takes the browser down too.
        try:
            # Try creating new page
            if context and len(context.pages) > 0:
                try:
                    page = context.new_page()
                    break
                except Exception as e:
                    print(f"Failed to create page: {e}")
                    context = None
            
            # No context, or the page could not be created: reconnect
            if not context or not page:
                print(f"Reconnecting to browser... (attempt {retry_count + 1}/{max_retries})")
                
                # Try disconnecting existing connection first
                try:
                    if browser:
                        browser.close()
                except:
                    pass
                
                # Wait for connection to fully close
                import time
                time.sleep(2)
                
                # Reconnect
                try:
                    cdp_h = get_cdp_headers(); browser = p.chromium.connect_over_cdp(cdp_url, **({"headers": cdp_h} if cdp_h else {}))
                    # Wait for connection to stabilize
                    time.sleep(1)
                    
                    # Get or create context
                    if browser.contexts:
                        context = browser.contexts[0]
                    else:
                        context = browser.new_context()
                    
                    page = context.new_page()
                    break
                    
                except Exception as e:
                    print(f"Connection failed: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise Exception(f"Cannot connect to browser after {max_retries}  times")
                    continue
                    
        except Exception as e:
            print(f"Error creating page: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                raise
    
    if not page:
        raise Exception("Cannot create page")

    # Single-tab reset: the new page is ready, so closing the rest cannot drop the
    # page count to 0. It can be turned off with agent.single_tab_reset in config.json;
    # it is a kill switch, see the comment on RuntimeConfig.single_tab_reset.
    if get_config().single_tab_reset:
        n_closed = close_other_pages(page.context, keep=page)
        if n_closed:
            print(f"🧹 single-tab reset: closed {n_closed} leftover page(s)")
    else:
        print(f"⏭️ single-tab reset is off (single_tab_reset=false); {len(page.context.pages)} tab(s) open")

    # Set viewport size
    try:
        # Get browser window size
        browser_window_size = page.evaluate("""
            () => {
                return {
                    availWidth: window.screen.availWidth,
                    availHeight: window.screen.availHeight,
                    width: window.outerWidth || 1920,
                    height: window.outerHeight || 1080
                }
            }
        """)
        
        print(f"Available screen size: {browser_window_size['availWidth']}x{browser_window_size['availHeight']}")
        print(f"Browser window size: {browser_window_size['width']}x{browser_window_size['height']}")
        
        global browser_window_size_width
        global browser_window_size_height
        
        browser_window_size_width = browser_window_size['width']
        browser_window_size_height = browser_window_size['height']
        
        page.set_viewport_size({"width": browser_window_size_width, "height": browser_window_size_height})
    except Exception as e:
        print(f"Setting the viewport size failed: {e}; using the default")
        browser_window_size_width = 1920
        browser_window_size_height = 1080
        page.set_viewport_size({"width": 1920, "height": 1080})
    
    # Add no-cache headers to every request from the page
    def prevent_cache(route):
        try:
            headers = route.request.headers.copy()
            headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            headers['Pragma'] = 'no-cache'
            headers['Expires'] = '0'
            route.continue_(headers=headers)
        except:
            route.continue_()
    
    # Install the routing rule
    try:
        page.route('**/*', prevent_cache)
    except Exception as e:
        print(f"Failed to set the routing rule: {e}")
    
    # Navigate to the target URL
    try:
        page.goto(target_url, wait_until="domcontentloaded", timeout=150000)
        waited = settle_after_navigation(page)
        print(f"⏱️ first-paint settle {waited}ms")
    except Exception as e:
        print(f"Navigation failed: {e}")
        # Try a looser wait condition
        try:
            page.goto(target_url, wait_until="commit", timeout=60000)
        except:
            # Still failing: at least the page exists
            pass
    
    return page

# Save a screenshot
def save_screenshot(page, savePath, timeout_ms=5000, max_retries=3):
    """
    Save a screenshot, with retries.
    """
    # Make sure the directory exists (only needed once)
    try:
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
    except Exception as e:
        print(f"### failed to create directory: {str(e)}")
        return False

    for i in range(max_retries):
        try:
            # Take the screenshot
            page.screenshot(path=savePath, full_page=False, timeout=timeout_ms)
            print(f"### screenshot saved to: {savePath}")
            return True
        except Exception as e:
            print(f"### screenshot failed (attempt {i+1}/{max_retries}): {str(e)}")
            if i < max_retries - 1:
                time.sleep(1)  # wait a moment before retrying
    
    return False

# CDP mouse move
def cdp_mouse_move(client, end_x, end_y, steps=20):
    """
    Emulate Playwright's multi-step page.mouse.move over the CDP protocol.
    
    params:
    client - CDP session
    end_x, end_y - target coordinates
    steps - number of intermediate moves
    """
    import time
    
    # Get the current mouse position first (assume a start position if that fails)
    try:
        # CDP has no direct way to read the mouse position, so ask JavaScript
        position = client.send("Runtime.evaluate", {
            "expression": "({x: window.mouseX || 0, y: window.mouseY || 0})",
            "returnByValue": True
        })
        start_x = position["result"]["value"]["x"]
        start_y = position["result"]["value"]["y"]
    except:
        # Unknown: assume the mouse starts at (0,0)
        start_x, start_y = 0, 0
    
    # Per-step increments
    delta_x = (end_x - start_x) / steps
    delta_y = (end_y - start_y) / steps
    
    # Move in steps
    for step in range(1, steps + 1):
        current_x = start_x + delta_x * step
        current_y = start_y + delta_y * step
        
        # Send the mouse-move event
        client.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved", 
            "x": current_x, 
            "y": current_y
        })
        
        # Optional: a short delay makes the movement look more natural
        time.sleep(0.01)  # 10 ms
    
    # Remember the mouse position for the next call
    client.send("Runtime.evaluate", {
        "expression": f"window.mouseX = {end_x}; window.mouseY = {end_y};"
    })

# CDP click
def mouse_up_and_down(page, client, x, y, time_wait=500):
    # client.send("Input.dispatchMouseEvent", {
    #     "type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1
    # })
    # page.wait_for_timeout(time_wait)  # hold the button down
    # client.send("Input.dispatchMouseEvent", {
    #     "type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1
    # })
    page.mouse.click(x, y)

# Click
def click(page, client, x, y, time_wait=500):
    """Glide the mouse there over CDP, click with Playwright, then give the page 3 s to react."""
    cdp_mouse_move(client, x, y, steps=20)
    #page.wait_for_timeout(500)

    # Perform the click
    mouse_up_and_down(page, client, x, y, time_wait)
    page.wait_for_timeout(3000)

    return page, client

# Double click
def doubleclick(page, client, x, y, time_wait=500):
    # Perform the clicks
    cdp_mouse_move(client, x, y, steps=20)
 
    time_wait = random.randint(time_wait // 2, time_wait)
    mouse_up_and_down(page, client, x, y, time_wait)
    
    time_wait = random.randint(time_wait // 2, time_wait)
    mouse_up_and_down(page, client, x, y, time_wait)

    page.wait_for_timeout(3000)
    
    return page, client

# Hotkeys
# Browser-UI shortcuts: CDP's Input.dispatchKeyEvent only injects keys into the
# *page's renderer process* and cannot reach Chrome's own UI, so these keys are
# always no-ops, on every platform. Letting them fail silently is far more dangerous
# than rejecting them: the model gets no negative feedback and keeps retrying until
# the steps run out. In an early 10-task run, 100 of 335 actions were hotkeys, 87 of
# them of this kind, and 8 of the 10 tasks idled to the step limit with no answer.
BROWSER_LEVEL_HOTKEYS = {
    "ctrl+l", "alt+d", "meta+l",              # focus the address bar
    "f5", "ctrl+r", "meta+r", "ctrl+shift+r",  # reload
    "f6", "f11", "f12",                        # focus cycling / full screen / devtools
    "ctrl+shift+i", "ctrl+shift+j", "ctrl+shift+c", "meta+alt+i", "meta+alt+j",
    "ctrl+t", "meta+t", "ctrl+w", "meta+w", "ctrl+shift+t",  # tab management
    "ctrl+n", "meta+n", "ctrl+shift+n",
    "ctrl+tab", "ctrl+shift+tab",
    "alt+left", "alt+right", "meta+left", "meta+right",       # history back / forward
    "ctrl+d", "meta+d", "ctrl+h", "ctrl+j", "ctrl+shift+delete",
}


class UnsupportedHotkey(Exception):
    """The model asked for a shortcut that CDP cannot deliver to the browser UI."""


def hotkey(page, client, hot_key_value, time_wait=500):
    """Press a hotkey. Browser-level shortcuts (see BROWSER_LEVEL_HOTKEYS) raise
    UnsupportedHotkey here and the executor passes the reason to the model. Single keys
    go out as CDP key events, combinations through Playwright's keyboard.press.
    """
    # Map common key names to Playwright's names
    key_map = {
        "ctrl": "Control", "control": "Control",
        "alt": "Alt", "shift": "Shift",
        "meta": "Meta", "cmd": "Meta", "command": "Meta",
        "enter": "Enter", "return": "Enter",
        "tab": "Tab", "escape": "Escape", "esc": "Escape",
        "backspace": "Backspace", "delete": "Delete", "del": "Delete",
        "space": " ", "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
        "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
        # The model often writes bare arrow names ("alt+Down"); without these Playwright raises "Unknown key"
        "up": "ArrowUp", "down": "ArrowDown",
        "left": "ArrowLeft", "right": "ArrowRight",
        "home": "Home", "end": "End",
        "pageup": "PageUp", "pagedown": "PageDown",
        "pgup": "PageUp", "pgdn": "PageDown", "pgdown": "PageDown",
    }

    # Normalize, then check the block-list (order- and case-insensitive)
    _parts = [k.strip().lower() for k in (
        hot_key_value.split("+") if "+" in hot_key_value else hot_key_value.strip().split())]
    _norm = "+".join(_parts)
    _norm_sorted = "+".join(sorted(_parts))
    if _norm in BROWSER_LEVEL_HOTKEYS or _norm_sorted in {
            "+".join(sorted(k.split("+"))) for k in BROWSER_LEVEL_HOTKEYS}:
        raise UnsupportedHotkey(
            f"hotkey(key='{hot_key_value}') is a browser-UI shortcut and cannot be delivered "
            f"to the page. It did NOT run. There is no way to navigate by URL. "
            f"If you wanted to refresh the page, use reload(). Do not retry this key.")

    # CDP keyCodes (for CDP Input.dispatchKeyEvent)
    cdp_key_codes = {
        "Enter": 13, "Tab": 9, "Escape": 27, "Backspace": 8, "Delete": 46,
        " ": 32, "ArrowUp": 38, "ArrowDown": 40, "ArrowLeft": 37, "ArrowRight": 39,
        "Home": 36, "End": 35, "PageUp": 33, "PageDown": 34,
    }

    # Accepts "ctrl+a", "ctrl+shift+t", "ctrl a", "enter", ...
    if "+" in hot_key_value:
        keys = [k.strip() for k in hot_key_value.split("+")]
    else:
        keys = hot_key_value.strip().split()
    mapped = [key_map.get(k.lower(), k) for k in keys]

    # A single key with a CDP keyCode: send it directly over CDP (more reliable)
    if len(mapped) == 1 and mapped[0] in cdp_key_codes:
        key_name = mapped[0]
        key_code = cdp_key_codes[key_name]
        client.send("Input.dispatchKeyEvent", {
            "type": "keyDown",
            "windowsVirtualKeyCode": key_code,
            "code": key_name,
            "key": key_name,
        })
        client.send("Input.dispatchKeyEvent", {
            "type": "keyUp",
            "windowsVirtualKeyCode": key_code,
            "code": key_name,
            "key": key_name,
        })
    else:
        # Combinations go through Playwright's keyboard.press
        combo = "+".join(mapped)
        page.keyboard.press(combo)

    page.wait_for_timeout(time_wait)
    return page, client

# Typing
def click_type(page, client, x, y, content, time_wait=500):
    """Execute type(): optionally click the coordinates first (current callers pass -1 =
    no click), then select all, delete, and type character by character.
    Keys go to **whatever has focus**, which the previous click decided; to target an
    input box reliably, use fill().
    """
    # 1. Click via CDP
    if x != -1 and y != -1:
        page, client = click(page, client, x, y, time_wait) 
        page.wait_for_timeout(random.randint(100, 200))
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Command+A (Meta+A) on macOS
        page.keyboard.press("Meta+a")
    else:  # Windows and Linux
        # Control+A on Windows and Linux
        page.keyboard.press("Control+a")
    page.keyboard.press('Backspace')
    page.wait_for_timeout(2500)

    for char in content:
        page.keyboard.type(char, delay=random.randint(10, 50))

    # Pause 2 s afterwards
    page.wait_for_timeout(2000)

    return page, client

# Scroll
def scroll(page, client, delta_x, delta_y):
    viewport_size = page.viewport_size
    center_x = viewport_size["width"] / 2
    center_y = viewport_size["height"] / 2
    client.send("Input.dispatchMouseEvent", {
        "type": "mouseWheel",
        "x": center_x,
        "y": center_y,
        "deltaX": delta_x, # positive scrolls down, negative scrolls up
        "deltaY": delta_y  # positive scrolls down, negative scrolls up
    })
    return page, client

# Scroll at a given point (e.g. inside a menu)
def scroll_menu(page, client, x, y, delta_x, delta_y):
    print("x,y,delta_y:", x, y, delta_y)
    cdp_mouse_move(client, x, y, steps=20)
    client.send("Input.dispatchMouseEvent", {
        "type": "mouseWheel",
        "x": x,
        "y": y,
        "deltaX": delta_x,
        "deltaY": delta_y  # positive scrolls down, negative scrolls up
    })
    return page, client

# Drag
def drag(page, client, x1, y1, x2, y2, steps=20):
    """
    page - Playwright page
    client - CDP session
    x1, y1 - start coordinates
    x2, y2 - target coordinates
    steps - number of intermediate moves
    """
    import time
    
    # 1. Move the mouse to the start position
    cdp_mouse_move(client, x1, y1, steps=steps)
    page.wait_for_timeout(100)
    
    # 2. Press the left button
    client.send("Input.dispatchMouseEvent", {
        "type": "mousePressed",
        "x": x1,
        "y": y1,
        "button": "left",
        "clickCount": 1
    })
    page.wait_for_timeout(100)
    
    # 3. Move to the target in steps
    delta_x = (x2 - x1) / steps
    delta_y = (y2 - y1) / steps
    
    for step in range(1, steps + 1):
        current_x = x1 + delta_x * step
        current_y = y1 + delta_y * step
        
        client.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": current_x,
            "y": current_y,
            "button": "left",  # keep the button down while dragging
            "buttons": 1       # left button pressed
        })
        
        # A short delay makes the drag look more natural
        time.sleep(0.01)  # 10 ms
    
    # 4. Release at the target
    client.send("Input.dispatchMouseEvent", {
        "type": "mouseReleased",
        "x": x2,
        "y": y2,
        "button": "left",
        "clickCount": 1
    })
    
    page.wait_for_timeout(2000)  # let the drag take effect
    return page, client

# Wait
def wait(page, client, time):
    page.wait_for_timeout(time)
    return page, client

# Step accounting
def step_cost(action):
    """How many steps this action counts as. A compound action counts once per browser
    operation: a click_many that clicks 4 squares is 4 steps. The max_steps budget is
    charged the same way; otherwise 6 rounds of 16 points would look like 6 steps
    while the browser was operated 96 times. Every other action is 1."""
    if action.get("name") == "ClickMany":
        return max(1, min(len(action.get("points") or []), MAX_CLICK_MANY_POINTS))
    return 1


def parse_action_type(action_str):
    """Action type from the start of the action string. agent.predict() uses it to detect the end (Finish / Wait / CallUser)."""
    action_str = action_str.strip()
    if action_str.startswith('click_many('):
        return 'ClickMany'
    if action_str.startswith('click('):
        return 'LeftClick'
    elif action_str.startswith('doubleclick(') or action_str.startswith('left_double('):
        return 'DoubleClick'
    elif action_str.startswith('right_single('):
        return 'RightClick'
    elif action_str.startswith('hover('):
        return 'Hover'
    elif action_str.startswith('select('):
        return 'Select'
    elif action_str.startswith('drag('):
        return 'Drag'
    elif action_str.startswith('hotkey(') or action_str.startswith('hotkey '):
        return 'Hotkey'
    elif action_str.startswith('reload('):
        return 'Reload'
    elif action_str.startswith('back('):
        return 'Back'
    elif action_str.startswith('find_text('):
        return 'FindText'
    elif action_str.startswith('read_text('):
        return 'ReadText'
    elif action_str.startswith('fill('):
        return 'Fill'
    elif action_str.startswith('type('):
        return 'Type'
    elif action_str.startswith('stop('):
        return 'Stop'
    elif action_str.startswith('scroll'):
        # current action-space format
        if "scroll(" in action_str: 
            if "up" in action_str:
                return 'ScrollUp'
            elif "down" in action_str:
                return 'ScrollDown'
            elif "left" in action_str:
                return 'ScrollLeft'
            elif "right" in action_str:
                return 'ScrollRight'
            else:
                print("scroll parse error !")
                exit()
        else:
            print("scroll parse error !")
            exit()
    elif action_str.startswith('goto('):
        return 'Goto'
    elif action_str.startswith('finished(') or action_str.startswith('finish('):
        return 'Finish'
    elif action_str == 'wait()':
        return 'Wait'
    elif action_str == 'call_user()':
        return 'CallUser'
    else:
        return 'Unknown'
           
# Parse an action
def parse_action(action_str):
    """Parse the model's action string into the executor's dict: {"name": ..., "coordinate" / "value" / ...}.

    excute_action() dispatches only on the name returned here. A new action needs a
    branch both here and in parse_action_type() (two of the three places that must
    line up). A parse failure always returns {"name": "Unknown"} and never raises.
    """
    response = action_str
    # print(response)
    try:
        if "click_many(" in response:
            # Both parse functions must know it (the lesson from back()). Coordinates were already mapped to page pixels in agent.py.
            m = re.search(r"points='([^']*)'", response)
            body = m.group(1) if m else response
            pts = [[int(x), int(y)] for x, y in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", body)]
            action = {"name": "ClickMany", "points": pts} if pts else {"name": "Unknown"}

        elif "type(content" in response:
            type_value = extract_between_keywords(response,"content='", "')")
            if len(type_value) == 0:
                type_value = extract_between_keywords(response, 'content="', '")')
            if len(type_value) == 0:
                print("type: could not parse the content!")
                exit()

            action = {"name": "Type", "value": type_value[0]}

        elif "click(start_box=" in response:
            coord = extract_box_coordinate(response, "start_box")
            if coord:
                cx, cy = coord
                action = {"name": "LeftClick", "coordinate": [cx, cy]}
            else:
                action = {"name": "Unknown"}

        elif "left_double(start_box=" in response or "doubleclick(start_box=" in response:
            coord = extract_box_coordinate(response, "start_box")
            if coord:
                cx, cy = coord
                action = {"name": "DoubleClick", "coordinate": [cx, cy]}
            else:
                action = {"name": "Unknown"}

        elif "right_single(start_box=" in response:
            coord = extract_box_coordinate(response, "start_box")
            if coord:
                cx, cy = coord
                action = {"name": "RightClick", "coordinate": [cx, cy]}
            else:
                action = {"name": "Unknown"}

        elif "hover(start_box=" in response:
            coord = extract_box_coordinate(response, "start_box")
            if coord:
                cx, cy = coord
                action = {"name": "Hover", "coordinate": [cx, cy]}
            else:
                action = {"name": "Unknown"}

        elif "select(start_box=" in response:
            coord = extract_box_coordinate(response, "start_box")
            m = (re.search(r"option='([^']*)'", response)
                 or re.search(r'option="([^"]*)"', response))
            if coord and m:
                cx, cy = coord
                action = {"name": "Select", "coordinate": [cx, cy], "value": m.group(1)}
            else:
                action = {"name": "Unknown"}

        elif "fill(start_box=" in response:
            # Must be added to both parse functions. The executor dispatches on *this*
            # function's action["name"]; parse_action_type() is only used for counting.
            # When back() was added to the latter alone, all 6 back() calls the model made
            # fell into Unknown and the executor never ran once.
            # Position: coming after the `if "type(content" in response` branch is fine
            # (fill uses content= and never contains "type(content"), but it must come
            # *before* the loose substring matches (scroll / hotkey) or it may be swallowed.
            coord = extract_box_coordinate(response, "start_box")
            m = (re.search(r"content='(.*?)'\s*\)", response, re.S)
                 or re.search(r'content="(.*?)"\s*\)', response, re.S))
            if coord and m:
                cx, cy = coord
                action = {"name": "Fill", "coordinate": [cx, cy], "value": m.group(1)}
            else:
                action = {"name": "Unknown"}

        elif "scroll(" in response:
            # direction
            dir_match = re.search(r"direction='(\w+)'", response)
            direction = dir_match.group(1) if dir_match else None

            # coordinates (both the <|box_start|> and the (x,y) formats)
            coord = extract_box_coordinate(response, "start_box")

            if direction:
                direction_map = {"up": "ScrollUp", "down": "ScrollDown", "left": "ScrollLeft", "right": "ScrollRight"}
                action = {"name": direction_map.get(direction, "ScrollDown")}
            elif "up" in response:
                action = {"name": "ScrollUp"}
            elif "down" in response:
                action = {"name": "ScrollDown"}
            elif "left" in response:
                action = {"name": "ScrollLeft"}
            elif "right" in response:
                action = {"name": "ScrollRight"}
            else:
                print("scroll parse error !")
                exit()

            if coord:
                action["coordinate"] = [coord[0], coord[1]]

        elif "hotkey" in response:
            # Support both key='...' and key="..."
            hotkey_value = extract_between_keywords(response, "key='", "')")
            if not hotkey_value:
                hotkey_value = extract_between_keywords(response, "key=\"", "\")")
            if not hotkey_value:
                hotkey_value = extract_between_keywords(response, "\"", "\"")
            if hotkey_value:
                hotkey_value = hotkey_value[0].replace(".", "")
            else:
                hotkey_value = ""
            print(f"hotkey_value:{hotkey_value}")
            action = {"name": "HotKey", "value": hotkey_value}

        elif "drag" in response:
            start = extract_box_coordinate(response, "start_box")
            end = extract_box_coordinate(response, "end_box")
            if start and end:
                x1, y1 = start
                x2, y2 = end
                action = {"name": "Drag", "bbox": [x1, y1, x2, y2]}
            else:
                # Fall back to old format
                try:
                    click_point = extract_between_keywords(response, "start_box='(", ")'")[0]
                    c1, c2 = click_point[0].split("),(")
                    x1, y1 = c1.split(",")
                    x2, y2 = c2.split(",")
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    action = {"name": "Drag", "bbox": [x1, y1, x2, y2]}
                except:
                    action = {"name": "Unknown"}

        elif "reload(" in response:
            action = {"name": "Reload"}

        elif "read_text(" in response:
            # Must be added to both parse functions: the executor dispatches on *this*
            # function's action["name"]. Must come before find_text: both contain
            # "_text(", and in the other order this branch would be swallowed.
            m = re.search(r"read_text\(\s*near\s*=\s*(['\"])(.*?)\1\s*\)", response, re.S)
            if not m:
                m = re.search(r"read_text\(\s*(['\"])(.*?)\1\s*\)", response, re.S)
            action = {"name": "ReadText", "value": m.group(2)} if m else {"name": "Unknown"}

        elif "find_text(" in response:
            # Must be added to both parse functions. The executor dispatches on the
            # action["name"] returned by *this* function; parse_action_type() is only
            # used for counting (see the fill branch above).
            m = re.search(r"find_text\(\s*text\s*=\s*(['\"])(.*?)\1\s*\)", response, re.S)
            if not m:   # tolerate find_text('xxx')
                m = re.search(r"find_text\(\s*(['\"])(.*?)\1\s*\)", response, re.S)
            action = {"name": "FindText", "value": m.group(2)} if m else {"name": "Unknown"}

        elif "back(" in response:
            # Must be added to *this* function. The executor dispatches on
            # parse_action()'s action["name"], not on the string parse_action_type()
            # returns; they are different functions. When back() was first enabled it
            # was added to parse_action_type only, the model's 6 back() calls all fell
            # into the Unknown branch as "not executed", and the executor code never ran.
            # This is what "action space / parser / executor must line up" means.
            action = {"name": "Back"}

        elif "goto(" in response:
            m = re.search(r"goto\(\s*url\s*=\s*(['\"])(.*?)\1\s*\)", response, re.S)
            if not m:   # tolerate goto('https://...')
                m = re.search(r"goto\(\s*(['\"])(.*?)\1\s*\)", response, re.S)
            if m:
                action = {"name": "Goto", "value": m.group(2).strip()}
            else:
                action = {"name": "Unknown"}

        elif "finished(" in response or "finish(" in response:
            action = {"name": "Finish"}

        elif "call_user()" in response:
            action = {"name": "CallUser"}

        elif "wait()" in response:
            action = {"name": "Wait"}

        else:
            action = {"name": "Unknown"}

    except:
        print("parse error")
        action = {"name": "Unknown"}

    return action

# Execute actions
# Feedback on the last action. main.py takes it every step and passes it to the model
# (obs["note"] in agent.predict). Without this channel "the action had no effect" is
# invisible to the model: it only sees an unchanged screenshot and repeats the action
# until the steps run out (the most common failure in our first runs).
LAST_ACTION_NOTE = ""


def take_action_note():
    """Take and clear the feedback (consumed once, so the same note is never sent on two steps)."""
    global LAST_ACTION_NOTE
    note, LAST_ACTION_NOTE = LAST_ACTION_NOTE, ""
    return note


# ── No-op detection ────────────────────────────────────────────────────────
# In one development set, 47 of 70 tasks hit the step limit with no answer, and **46
# of them (97%)** repeated the same action 3 or more times; repeated actions used up
# 39% of all steps (736 of 1841). Same mechanism as the browser-hotkey idling fixed
# earlier: the action runs, the page does not react, the model sees an unchanged
# screenshot and tries again until the budget is gone.
#
# Why a DOM fingerprint and not a screenshot diff: real pages are full of clocks,
# carousels and ads that refresh every second (one target site shows a live clock at
# the top), so a pixel diff would report "changed" on every step.
# Why not ask an LLM to compare: the model already knows ("the click had no visible
# effect", "tried several times but it won't switch"). The bottleneck is not
# detection; it is that the model gets no hard feedback and retries anyway.
_FP_JS = """() => {
  try {
    return {
      url: location.href,
      title: document.title,
      nodes: document.getElementsByTagName('*').length,
      text: (document.body && document.body.innerText || '').length,
      sx: Math.round(window.scrollX), sy: Math.round(window.scrollY),
      // Many sites scroll an inner container, not the document; looking only at
      // window.scrollY would report a real scroll as "no effect". A false alarm is
      // worse than a miss: it talks the model out of a move that works.
      inner: Array.from(document.querySelectorAll('*')).reduce((a,e)=>a+(e.scrollTop|0),0),
      active: (document.activeElement && (document.activeElement.className || document.activeElement.tagName)) || '',
      // FNV-1a hash of the body text. **Not in _FP_KEYS**: a clock changes it every
      // second, so using it everywhere would switch off no-op detection on any page with
      // a clock. Used only for hovers and clicks on chart elements: a tooltip changing
      // from "Health services: 467" to "...: 502" keeps the same length and node count,
      // and only the hash sees it.
      thash: (function (t) { let h = 2166136261;
               for (let i = 0; i < t.length; i++) { h ^= t.charCodeAt(i); h = Math.imul(h, 16777619); }
               return h >>> 0; })(document.body && document.body.innerText || '')
    };
  } catch (e) { return null; }
}"""


def page_fingerprint(page):
    """Fingerprint of the page state: insensitive to clocks and animation, sensitive to real interaction. About 1 ms, no API cost."""
    try:
        return page.evaluate(_FP_JS)
    except Exception:
        return None


_DESCRIBE_POINT_JS = """([x,y]) => {
  const el = document.elementFromPoint(x, y);
  if (!el) return {tag: null};
  const cs = getComputedStyle(el);
  return {
    tag: el.tagName,
    cls: (el.className && String(el.className).slice(0,60)) || '',
    text: (el.innerText || '').trim().slice(0, 40),
    clickable: !!(el.onclick || el.closest('a,button,[role=button],[role=tab],input,select,textarea,label')),
    disabled: !!el.disabled || cs.pointerEvents === 'none'
  };
}"""


def describe_point(page, x, y):
    """What is actually under this point; tells in advance whether a click will miss.

    On an iframe it describes the element *inside* the frame: the main document's
    elementFromPoint only returns the `<iframe>` itself, and reporting that as
    "<IFRAME> is not clickable" would tell the model that a perfectly clickable tab
    (seen in an embedded chart) cannot be clicked.
    """
    hit = frame_at(page, x, y)
    frame, ox, oy = hit if hit else (page, 0.0, 0.0)
    try:
        info = frame.evaluate(_DESCRIBE_POINT_JS, [x - ox, y - oy])
    except Exception:
        return None
    if isinstance(info, dict) and hit is not None and _is_child_frame(frame):
        info["frame"] = True
    return info


# ── Clicks inside iframes ──────────────────────────────────────────────────
# Measured with an embedded chart (the same embed reproduced on a local file:// page):
#   1. The main-frame fingerprint is **blind** to changes inside an iframe: a tab
#      switch succeeded and the chart's y-axis went from 6,000 to 2,000, yet the main
#      frame's url / title / node count / text length did not move. The model was told
#      "the click had no effect, this is ineffective action #N, change strategy",
#      false feedback that talked it out of the right path.
#   2. With the tab scrolled to the bottom of the viewport (y ~ 917 of 1080), physical
#      CDP clicks were **swallowed three times in a row**; scrolled to the middle or
#      top, one click worked, and a DOM event dispatched inside the iframe worked at
#      once. In one task the agent found the right report by step 50 and spent the
#      remaining 30 steps on this.
# So when a click lands on an iframe, the fingerprint of **the child frame that was
# hit** decides whether it worked, and if the physical click had no effect a DOM click
# is dispatched inside that frame. The fallback only runs when neither the main frame
# nor the child frame changed: a click that worked is not touched, and clicks on the
# main frame never take this path.
# The page fingerprint is not the sum over all frames: carousels in ad iframes change
# every second and would make no-op detection useless (the scroll breaker relies on it).
def _is_child_frame(frame):
    try:
        return frame.parent_frame is not None
    except Exception:
        return False


def frame_at(page, x, y):
    """Which frame the page coordinates (x, y) fall on: (frame, ox, oy); the main frame is (main, 0, 0).

    Same rule as editable_at: Playwright's page.frames lists the main frame first and
    deeper frames later, and later hits override earlier ones, so the result is the
    (deepest) frame the model actually sees. bounding_box is relative to the main
    frame's viewport, so subtracting the offset once gives local coordinates.
    Any exception (a cross-origin frame mid-navigation, a test double without frames)
    returns None and callers treat it as "not in an iframe": this path may only add
    value, never break the click itself.
    """
    found = None
    try:
        for frame in page.frames:
            try:
                if frame.parent_frame is None:
                    found = (frame, 0.0, 0.0)
                    continue
                box = frame.frame_element().bounding_box()
                if not box:
                    continue
                if not (box["x"] <= x <= box["x"] + box["width"]
                        and box["y"] <= y <= box["y"] + box["height"]):
                    continue
                found = (frame, box["x"], box["y"])
            except Exception:
                continue
    except Exception:
        return None
    return found


def frame_fingerprint(frame):
    """A frame's own state fingerprint, using the same JS as page_fingerprint. None if unavailable."""
    try:
        return frame.evaluate(_FP_JS)
    except Exception:
        return None


#: No DOM-event fallback inside human-verification providers' iframes: they accept only
#: trusted events, so synthetic ones never work (4 fallback clicks on a Turnstile
#: checkbox all landed on <BODY> with no effect), and injecting synthetic events into
#: a CAPTCHA is exactly what bot detection looks for. The physical click still
#: happens; there is just no second attempt.
_CAPTCHA_FRAME_HINTS = ("challenges.cloudflare.com", "recaptcha", "hcaptcha", "captcha")


def child_frame_hit(page, x, y):
    """If the click lands inside an iframe, return (frame, ox, oy, fingerprint before the action); otherwise None.
    Frames of CAPTCHA providers also return None (see _CAPTCHA_FRAME_HINTS)."""
    hit = frame_at(page, x, y)
    if not hit or not _is_child_frame(hit[0]):
        return None
    frame, ox, oy = hit
    try:
        url = (frame.url or "").lower()
    except Exception:
        url = ""
    if any(h in url for h in _CAPTCHA_FRAME_HINTS):
        return None
    return frame, ox, oy, frame_fingerprint(frame)


# A complete press-release-click sequence with coordinates (clientX/Y are local to the
# frame): dispatching only click misses controls driven by mousedown; dispatching only
# pointer events misses old-style onclick handlers. Canvas charts (e.g. Tableau) receive
# synthetic events with the right coordinates, which they treat like real clicks.
_JS_CLICK_AT = """([x, y]) => {
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  const init = {bubbles: true, cancelable: true, composed: true, view: window,
                clientX: x, clientY: y, button: 0, buttons: 1};
  const fire = (type) => el.dispatchEvent(
      (type.startsWith('pointer') && window.PointerEvent) ? new PointerEvent(type, init)
                                                          : new MouseEvent(type, init));
  fire('pointerdown'); fire('mousedown');
  init.buttons = 0;
  fire('pointerup'); fire('mouseup'); fire('click');
  return {tag: el.tagName, text: (el.innerText || '').trim().slice(0, 40)};
}"""


def locate_text(page, text):
    """Find elements containing `text` in the main frame, then in every child frame.

    Returns (locator, in a child frame?, number of matches), or (None, False, 0).
    **Main frame first**: when the main frame has a match, behaviour is exactly as
    before; child frames are searched only when the main frame has none.
    Why search child frames: across several runs, find_text hit 10 times and missed
    18 times, and most misses were inside iframes (chart labels in an embedded chart,
    section titles in a filings viewer, a country list that lives entirely in an
    iframe, where the agent scrolled for 40 steps looking for "Peru").
    `page.get_by_text` only looks at the main document and reports "not found" for all
    of them, leaving the model to scroll screen by screen.
    """
    loc = page.get_by_text(text, exact=False)
    n = loc.count()
    if n:
        return loc, False, n
    try:
        frames = list(page.frames)
    except Exception:
        return None, False, 0
    for frame in frames:
        try:
            if frame.parent_frame is None:
                continue
            loc = frame.get_by_text(text, exact=False)
            n = loc.count()
            if n:
                return loc, True, n
        except Exception:      # cross-origin frame navigating / detached: skip
            continue
    return None, False, 0


#: Elements in child frames are scrolled into view with JS: Playwright's
#: scroll_into_view_if_needed waits until the timeout without moving for elements **in
#: a cross-origin iframe that is itself outside the viewport** (measured: 3 s timeout
#: in both directions), while element.scrollIntoView scrolls the parent document too
#: and gets there within a second (browsers allow scrollIntoView across origins; only
#: focus is restricted). Main-frame elements still use Playwright's method, unchanged.
_SCROLL_INTO_VIEW_JS = "e => e.scrollIntoView({block: 'center', inline: 'nearest'})"


def js_click_in_frame(frame, lx, ly):
    """Click the element at (lx, ly) inside a child frame with DOM events. Returns a description of it, or None."""
    try:
        return frame.evaluate(_JS_CLICK_AT, [lx, ly])
    except Exception:
        return None


# The same fallback for hover: an iframe at the bottom of the viewport does not even
# receive mouseMoved (measured: with a bar at y ~ 956 no tooltip appeared; scrolled to
# y ~ 440 it appeared at once). Chart libraries hang their tooltips on mousemove /
# pointermove, so dispatching move events with coordinates inside the frame draws the
# tooltip into the screenshot.
_JS_HOVER_AT = """([x, y]) => {
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  const init = {bubbles: true, cancelable: true, composed: true, view: window,
                clientX: x, clientY: y, button: 0, buttons: 0};
  const fire = (type) => el.dispatchEvent(
      (type.startsWith('pointer') && window.PointerEvent) ? new PointerEvent(type, init)
                                                          : new MouseEvent(type, init));
  fire('pointerover'); fire('pointerenter'); fire('mouseover'); fire('mouseenter');
  fire('pointermove'); fire('mousemove');
  return {tag: el.tagName, text: (el.innerText || '').trim().slice(0, 40)};
}"""


def js_hover_in_frame(frame, lx, ly):
    """"Move the mouse" onto the element at (lx, ly) inside a child frame with DOM events. Returns a description, or None."""
    try:
        return frame.evaluate(_JS_HOVER_AT, [lx, ly])
    except Exception:
        return None


_NO_CHANGE_STREAK = [0]      # consecutive no-change actions (escalates the wording)
_NO_CHANGE_TOTAL = [0]       # no-change actions in this task (for the main loop's breaker)

# Per-task reload budget (MAX_RELOADS_PER_TASK below). Without a cap reload becomes a
# new way to idle: given an action that looks like work but makes no progress, the
# model repeats it until the steps run out (the ineffective hotkeys did exactly that).
# Why 2: when a site failed to load, one retry is usually enough; if two in a row do
# not help, the problem is on the site's side and more reloads will not fix it.
def sanitize_page_text(text):
    """Replace characters obfuscated by an anti-scraping font with `?`. Returns
    (cleaned text, number of characters replaced).

    Some sites map digits to **unassigned code points** and render them with a custom
    font: the page shows `1098.24`, while the DOM holds `𘜂𘜃𘜉𘜄𘜆𘛾𘜅` (U+18702...).
    `unicodedata.name()` raises ValueError for these code points, and normal text has
    none of them (0.0% unnamed characters on the data sites we checked).

    **Why clean character by character instead of judging a whole passage**: a line
    is usually "mostly normal text plus a few obfuscated digits". On one site a
    225-character line had only 7 obfuscated characters (3.5%), so any reasonable
    whole-passage threshold would let it through, and the obfuscated part was exactly
    the number we needed. All-or-nothing loses both ways: let it through and the model
    takes `𘜂𘜃𘜉` as the answer (worse than not reading at all); reject the passage and
    the useful context (title, author, date) goes with it.
    Replacing per character and telling the model how many were replaced lets it judge
    for itself which part cannot be trusted.

    ⚠️ This only covers this one technique. When a value is drawn on a canvas, injected
    with CSS ::before or is an image, the DOM simply does not have it (rather than
    having garbage); read_text then returns nothing, which is a safe way to fail.
    """
    out, bad = [], 0
    for c in str(text):
        if c.isspace():
            out.append(c)
            continue
        try:
            unicodedata.name(c)
            out.append(c)
        except ValueError:
            out.append("?")
            bad += 1
    return "".join(out), bad


def looks_obfuscated(text, threshold=0.05):
    """Whether a passage is mostly obfuscated. Kept for "this passage is basically unusable";
    for anything finer use sanitize_page_text (see the 3.5% case in its docstring)."""
    chars = [c for c in str(text) if not c.isspace()]
    if not chars:
        return False
    _, bad = sanitize_page_text("".join(chars))
    return bad / len(chars) > threshold


MAX_RELOADS_PER_TASK = 2
_RELOAD_COUNT = [0]

# Two gates on click_many: at most 16 points per call (a 4x4 grid) and 6 calls per task
# (reCAPTCHA usually takes 2-3 rounds). Beyond that it is rejected with a reason: it
# exists for CAPTCHAs and must not become a general way to batch actions without looking.
MAX_CLICK_MANY_POINTS = 16
MAX_CLICK_MANY_PER_TASK = 6
_CLICK_MANY_COUNT = [0]

# Per-task back() budget, for the same reason (asking does not work; the executor
# has to enforce it). See runtime_config.max_backs_per_task for the value.
_BACK_COUNT = [0]

# Per-task find_text budget, for the same reason.
_FIND_TEXT_COUNT = [0]

# The text of the last find_text and which match it jumped to (0-based). Searching the
# same text again jumps to the next match. In one task the model called
# find_text('Commitments and Contingencies') three times and landed on the first match
# (a passing mention under Risk Factors) every time; it wrote "there are 4 occurrences
# on the page, I need to get to a later one", had no action that could do it, and
# ended up reading from another table. In long documents the first match is often the
# table of contents or a reference in the text; the real table comes later.
_FIND_LAST = ["", 0]

# Per-task read_text budget. Tighter than find_text: every call pushes text into the
# context, and too much dilutes attention (full-page text measured 2.6k-6.3k
# characters, on the order of a screenshot).
_READ_TEXT_COUNT = [0]


def reset_stall_counters():
    """Reset at the start of every task."""
    _NO_CHANGE_STREAK[0] = 0
    _NO_CHANGE_TOTAL[0] = 0
    _RELOAD_COUNT[0] = 0
    _BACK_COUNT[0] = 0
    _FIND_TEXT_COUNT[0] = 0
    _FIND_LAST[0], _FIND_LAST[1] = "", 0
    _CLICK_MANY_COUNT[0] = 0
    _READ_TEXT_COUNT[0] = 0


def reload_count():
    return _RELOAD_COUNT[0]


def back_count():
    return _BACK_COUNT[0]


def find_text_count():
    return _FIND_TEXT_COUNT[0]


def read_text_count():
    return _READ_TEXT_COUNT[0]


def no_change_streak():
    """Number of **consecutive** no-change actions (from fingerprint comparison).

    The main loop's scroll breaker uses it to tell apart two things that look alike:
      - truly idle: scrolling again and again while the fingerprint never changes
      - reading a long table screen by screen: also scrolling in a row, but every
        screen differs and the streak resets
    Counting scrolls alone cannot tell them apart and once cut off a task that was
    reading a table (see the comment in main.py).
    """
    return _NO_CHANGE_STREAK[0]


def stall_count():
    return _NO_CHANGE_TOTAL[0]


_FP_KEYS = ("url", "title", "nodes", "text", "sx", "sy", "inner", "active")


def _fp_changed(before, after, ignore=(), text_hash=False):
    """Whether two fingerprints differ. If either is missing (None) it counts as **changed**:
    a false "no change" is worse than a missed one, because it talks the model out of a
    move that works. Keys in `ignore` are not compared; with `text_hash=True` a change
    in the body-text hash also counts (see thash in _FP_JS; hovers and chart elements only)."""
    if not before or not after:
        return True
    if not isinstance(before, dict) or not isinstance(after, dict):
        return before != after
    if any(before.get(k) != after.get(k) for k in _FP_KEYS if k not in ignore):
        return True
    return bool(text_hash) and before.get("thash") != after.get("thash")


#: A click or hover on these elements means the agent is reading a chart: a tooltip
#: appearing or changing often leaves node count and text length unchanged, so the
#: no-op test must look at the body-text hash. SVG tagNames are lower case ('path'),
#: HTML ones upper case ('DIV'), which tells them apart.
_CHART_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "g",
               "svg", "text", "tspan", "image", "use", "CANVAS"}


def _diff_note(before, after, action_desc, point_info=None, ignore=(), text_hash=False):
    """Compare fingerprints. If nothing changed, return actionable feedback; if something did, reset the streak."""
    if not before or not after:
        return ""
    if _fp_changed(before, after, ignore, text_hash):
        _NO_CHANGE_STREAK[0] = 0
        return ""
    _NO_CHANGE_STREAK[0] += 1
    _NO_CHANGE_TOTAL[0] += 1
    bits = [f"{action_desc} produced NO detectable change "
            f"(url, title, DOM node count, text length, scroll offsets and focus are all identical)."]
    if point_info is not None:
        if point_info.get("tag") is None:
            bits.append("Nothing is rendered at that coordinate - the click hit empty space.")
        else:
            bits.append(
                f"The element at that point is <{point_info.get('tag')} class=\"{point_info.get('cls','')}\">"
                f" text=\"{point_info.get('text','')}\""
                + (", which is NOT clickable" if not point_info.get("clickable") else "")
                + (" and is disabled/pointer-events:none" if point_info.get("disabled") else "")
                + (" (it sits inside an embedded frame)" if point_info.get("frame") else "") + ".")
    if _NO_CHANGE_STREAK[0] >= 2:
        bits.append(f"This is {_NO_CHANGE_STREAK[0]} ineffective actions in a row. "
                    "Stop repeating - change approach entirely: target a different element, "
                    "scroll to bring the control into view, or try a different element.")
    else:
        bits.append("Do not repeat the same action - try a different element or approach.")
    return " ".join(bits)


def select_option_at(page, x, y, option):
    """Find the native <select> at the coordinates and choose the matching option.

    Why a dedicated action: a native <select> opens an **OS-level popup** that is not
    part of the page's render tree. It never shows in the screenshot and cannot be hit
    with page coordinates. One task died exactly here: the model clicked the same spot
    9 times, then tried a double click and Tab + typing, all without effect, until all
    40 steps were gone.
    Custom div-based dropdowns are unaffected (their options are drawn on the page);
    just click them.

    Returns (success?, feedback for the model). On failure the <select>'s **list of
    options** goes back to the model: it cannot see the popup, so guessing option names
    is all it can do, and handing it the list is far cheaper than letting it guess.
    """
    handle = page.evaluate_handle(
        "([x, y]) => { const el = document.elementFromPoint(x, y);"
        "               return el && el.closest('select'); }",
        [x, y])
    el = handle.as_element()
    if el is None:
        return False, ("There is no native <select> at that point, so select() did nothing. "
                       "If this is a custom dropdown built out of page elements, just click "
                       "it - its options are drawn on the page and can be clicked directly.")
    opts = page.evaluate(
        "(el) => Array.from(el.options).map(o => ({text: (o.text || '').trim(), value: o.value}))",
        el)
    if not opts:
        return False, "That <select> has no options yet - the page may still be loading."

    # Option text often differs from how people (and tasks) write it: a country list
    # says "Viet Nam" while the task and the model say "Vietnam". Without normalization
    # nothing matches, and since the model cannot see the popup it cannot learn the
    # right spelling and just retries.
    # Normalization = letters and digits only, lower case: "Viet Nam" / "vietnam" /
    # "VIET-NAM" all match.
    def norm(t):
        return re.sub(r"[^0-9a-z]+", "", (t or "").lower())

    want, want_n = (option or "").strip().lower(), norm(option)
    idx = None
    for i, o in enumerate(opts):                       # 1) exact match
        if o["text"].lower() == want or (o["value"] or "").lower() == want:
            idx = i
            break
    if idx is None and want_n:                         # 2) exact match after normalization
        for i, o in enumerate(opts):
            if norm(o["text"]) == want_n or norm(o["value"]) == want_n:
                idx = i
                break
    if idx is None and want_n:                         # 3) substring match after normalization
        for i, o in enumerate(opts):
            if want_n in norm(o["text"]):
                idx = i
                break
    if idx is None:
        preview = " | ".join(o["text"] for o in opts[:25])
        more = f" (+{len(opts) - 25} more)" if len(opts) > 25 else ""
        return False, (f"No option matching '{option}' in that dropdown. "
                       f"Its options are: {preview}{more}")

    el.select_option(index=idx)
    return True, ""


#: What counts as an editable element. Excludes inputs that can be clicked but not
#: filled (checkbox / radio / button): when the model points at one of those, we say
#: plainly "this is not a text box" instead of raising a Playwright exception.
_EDITABLE_SELECTOR = (
    "input:not([type=hidden]):not([type=checkbox]):not([type=radio])"
    ":not([type=submit]):not([type=button]):not([type=file]):not([disabled]),"
    "textarea:not([disabled]),"
    "[contenteditable]:not([contenteditable=false])"
)

_EDITABLE_AT_JS = """([x, y, sel]) => {
    const el = document.elementFromPoint(x, y);
    if (!el) return null;
    const contains = (n) => {
        const r = n.getBoundingClientRect();
        return r.width > 0 && r.height > 0 &&
               x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
    };
    const pick = (root, requireContain) => {
        const list = Array.from(root.querySelectorAll(sel));
        const hit = list.find(contains);
        if (hit) return hit;
        // With a single candidate, do not require containment: the model may point at the border or padding.
        if (!requireContain && list.length === 1) return list[0];
        return null;
    };
    // 1) Pointing at the input itself, or at a node inside it
    const up = el.closest(sel);
    if (up) return up;
    // 2) Pointing at a container *around* the input (common for search boxes with borders/icons)
    const down = pick(el, false);
    if (down) return down;
    // 3) Pointing at an overlay or icon *covering* the input: the input is a sibling,
    //    out of reach of both closest and querySelector. This is the shape of "the box
    //    has focus but the click produced no detectable change".
    //    Walk up to the containers, but **only accept a candidate whose rectangle really
    //    covers the point**: a form usually has several fields, and filling the wrong
    //    one is worse than filling none.
    let node = el.parentElement;
    for (let i = 0; i < 4 && node; i++, node = node.parentElement) {
        const sib = pick(node, true);
        if (sib) return sib;
    }
    return null;
}"""


def editable_at(page, x, y):
    """Find the editable element at page coordinates (x, y), descending into iframes if
    needed. None if there is none.

    Why iframes: in the main document `document.elementFromPoint` returns only **the
    `<iframe>` element itself** and cannot see inside. Many of the sites we targeted
    use iframes heavily; in one task the model said "the search box is an iframe and
    cannot be clicked directly" and ran out of all 40 steps.

    Playwright's `bounding_box()` is relative to the **main frame's viewport**, so
    however deeply an iframe is nested, subtracting its offset once gives the frame's
    local coordinates; no recursive accumulation is needed. `page.frames` lists the main
    frame first and descendants after it, deeper ones later, so later hits override
    earlier ones and the result is the deepest frame, the one the model actually sees.
    """
    found = None
    for frame in page.frames:
        try:
            if frame.parent_frame is None:
                ox, oy = 0.0, 0.0
            else:
                fe = frame.frame_element()
                box = fe.bounding_box()
                if not box:
                    continue
                if not (box["x"] <= x <= box["x"] + box["width"]
                        and box["y"] <= y <= box["y"] + box["height"]):
                    continue
                ox, oy = box["x"], box["y"]
            handle = frame.evaluate_handle(_EDITABLE_AT_JS, [x - ox, y - oy, _EDITABLE_SELECTOR])
            el = handle.as_element()
            if el is not None:
                found = el
        except Exception:
            # Cross-origin or navigating frames raise. Skip them; the main frame was already tried.
            continue
    return found


#: Native date inputs (<input type=date> and relatives) **accept only ISO format**:
#: typing "10/31/2019" key by key leaves the box empty (confirmed in a real browser),
#: while tasks and the model both write M/D/YYYY. So these get special handling:
#: convert to ISO and set the value with the element-level fill(); if the conversion
#: fails, tell the model exactly which format the box wants so it can rewrite the
#: value, instead of leaving it to guess at an empty box.
_DATE_INPUT_TYPES = {"date": "YYYY-MM-DD", "month": "YYYY-MM", "week": "YYYY-Www",
                     "time": "HH:MM", "datetime-local": "YYYY-MM-DDTHH:MM"}


def _to_iso_date(text):
    """M/D/YYYY, YYYY-M-D, YYYY/M/D -> YYYY-MM-DD. None if not recognized."""
    t = (text or "").strip()
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", t)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
    m = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", t)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
    return None


def split_submit_newline(content):
    """Split off a trailing newline -> (press Enter?, text to type).

    The action space says "end with \n to submit", and the model often copies that as
    **two literal characters** (backslash + n): 79 of 356 type() calls (22%) had this
    shape. Occasionally it sends a real newline. Both must be recognized.

    The old type() path handled neither: `click_type` typed character by character and
    entered `\\` and `n` into the search box as text (e.g.
    `type(content='permit search\n')`), and **Enter was never pressed**. Same family as
    `<|eot_id|>`: the prompt promised behaviour the executor never implemented.
    """
    if content.endswith("\\n"):          # two literal characters, the model's most common form
        return True, content[:-2]
    if content.endswith("\n"):           # a real newline
        return True, content[:-1]
    return False, content


def fill_at(page, x, y, content, timeout_ms=3000):
    """Fill text into the input box at the coordinates and **report what the box ends up
    containing**. Returns (success?, feedback for the model).

    Why not click() + type(): type() carries no coordinates (`excute_action` hard-codes
    `cx, cy = -1, -1`), so ctrl+a / Backspace / keystrokes all go to **whatever has
    focus**. Whether that is the right box depends entirely on the previous click, and
    a slightly missed click **is invisible in the screenshot**: in one development set
    15 of 70 tasks typed exactly the same content more than once, and one task failed
    once on each of its four fields.

    Why keep real keystrokes instead of `el.fill(content)`: setting the value in one go
    fires a single input event, and **suggestion / autocomplete dropdowns never open**,
    while search and filter boxes are exactly what we type into. So only **locating and
    clearing** move to element level (those are the steps that fail silently); the
    keystrokes stay as they were.

    Why read back: the model has no other way to know. On failure it only sees an
    unchanged screenshot and repeats itself; and a date box rewrites `10/31/2019` as
    `2019-10-31`, so reading back tells it the value did go in, just displayed in a
    different format, and saves a second attempt.
    """
    el = editable_at(page, x, y)
    if el is None:
        return False, ("There is no text box at that point, so fill() did nothing and the page "
                       "did not change. Point at the box itself (the area you would click into "
                       "before typing). If this control is not a text box at all, click it "
                       "instead; if the page takes keystrokes without any box, use type().")

    submit, text = split_submit_newline(content)

    try:
        el.scroll_into_view_if_needed(timeout=timeout_ms)
    except Exception:
        pass                                    # failing to scroll does not mean it cannot be filled

    try:
        el.click(timeout=timeout_ms)
    except Exception:
        # When an overlay covers the box, Playwright's actionability check times out.
        # focus() skips that check, which is enough for filling in a value; the
        # model reading the screenshot is the step that needs the box to be visible.
        try:
            el.focus()
        except Exception as e:
            return False, (f"That text box could not be focused ({str(e)[:120]}). Something may "
                           f"be covering it - close any popup or cookie banner first.")

    # Native date input: keystrokes do not work, so use ISO + element-level fill().
    itype = ""
    try:
        itype = (page.evaluate("(el) => (el.getAttribute('type') || '').toLowerCase()", el) or "")
    except Exception:
        pass
    if itype in _DATE_INPUT_TYPES:
        want = _to_iso_date(text) if itype == "date" else text
        if want is None:
            return False, (f"That is a date box - it only accepts "
                           f"{_DATE_INPUT_TYPES[itype]} (for example 2019-10-31). Nothing was "
                           f"typed. Call fill() again with the date written that way.")
        try:
            el.fill(want)
        except Exception as e:
            return False, (f"That date box did not accept '{want}' ({str(e)[:100]}). It wants "
                           f"{_DATE_INPUT_TYPES[itype]}.")
        page.wait_for_timeout(400)
        got = page.evaluate("(el) => el.value", el)
        if submit:
            page.keyboard.press("Enter")
            page.wait_for_timeout(800)
        tail = " Enter was pressed." if submit else ""
        return True, (f"The date box now contains '{got}'.{tail} Date boxes always display "
                      f"{_DATE_INPUT_TYPES[itype]} internally even if the page shows another "
                      f"format - read the screenshot to confirm it is the date you wanted.")

    try:
        el.fill("")                             # element-level clear: dispatches an input event
    except Exception:
        # Custom components may reject fill(). Fall back to select-all + delete on the
        # keyboard; it at least acts on the element we just focused, which still beats
        # the old path.
        page.keyboard.press("Control+a")
        page.keyboard.press("Backspace")

    if text:
        page.keyboard.type(text, delay=random.randint(10, 30))
    page.wait_for_timeout(600)                  # let suggestion lists / validation callbacks render

    got = None
    try:
        got = page.evaluate(
            "(el) => (el.value !== undefined && el.value !== null) ? el.value : el.innerText", el)
    except Exception:
        pass

    if submit:
        page.keyboard.press("Enter")
        page.wait_for_timeout(800)

    tail = " Enter was pressed." if submit else ""
    if got is None:
        return True, (f"Typed '{text[:60]}' into that box.{tail} Check the screenshot to confirm "
                      f"it landed where you wanted.")
    if got.strip() == text.strip():
        return True, f"The box now contains '{got[:60]}'.{tail}"
    # A mismatch is not necessarily a failure: date boxes reformat and suggestion boxes complete. Report the real value and let the model judge.
    return True, (f"Typed '{text[:60]}', and the box now contains '{got[:60]}'. If that is not "
                  f"what you wanted, the box may reformat or auto-complete what you type - "
                  f"read the screenshot before deciding.{tail}")


def excute_action(page, client, action):
    """Execute one action. Returns (page, client), the page possibly having switched.

    The same sequence every time:
      1. Record the page fingerprint and, for click-like actions, what is under the point
      2. Dispatch on action["name"]; a rejected or failed action writes its reason to
         LAST_ACTION_NOTE
      3. If the action opened a new tab, switch to it (closing old pages, keeping at least one)
      4. Wait for rendering to settle
      5. For actions without a note, compare fingerprints: if the page did not change,
         produce "do not repeat this" feedback
    """
    global LAST_ACTION_NOTE
    LAST_ACTION_NOTE = ""
    name = action["name"]

    # All pages before the action
    old_pages = set(page.context.pages)

    # No-op detection: fingerprint before the action + what is under the point for click-like actions
    _fp_before = page_fingerprint(page)
    _point = None
    _frame_hit = None      # click/hover inside an iframe: (child frame, offset x, offset y, fingerprint before)
    _text_hash = False     # hover / chart element: also compare the body-text hash (same-length tooltip swaps)
    if name in ("LeftClick", "DoubleClick", "RightClick", "Hover") and "coordinate" in action:
        _point = describe_point(page, *action["coordinate"])
        _text_hash = (name == "Hover"
                      or bool(_point and _point.get("tag") in _CHART_TAGS))
        if name != "RightClick" and get_config().enable_iframe_click_fallback:
            _frame_hit = child_frame_hit(page, *action["coordinate"])

    viewport_size = page.viewport_size
    page_height = viewport_size["height"]
    page_width = viewport_size["width"]

    if name == "Type":
        content = action["value"]
        cx,cy = -1,-1
        # A trailing "\n" should press Enter rather than be typed as text (see
        # split_submit_newline). This changes behaviour on **a path that passing tasks also
        # take**, so it has its own switch, separate from fill().
        submit = False
        if get_config().enable_type_newline_submit:
            submit, content = split_submit_newline(content)
        page, client = click_type(page, client, cx, cy, content, time_wait=1000)
        if submit:
            page.keyboard.press("Enter")
            page.wait_for_timeout(1500)
            # **Deliberately no LAST_ACTION_NOTE here**: the no-op check at the end of
            # excute_action only reports "the page did not change" when there is no note
            # (`if not new_pages and not LAST_ACTION_NOTE`). Taking that channel for a
            # successful Enter would deprive the model of the "nothing happened" signal,
            # the same trap as the ineffective hotkeys. The model can see the effect of
            # Enter in the screenshot.

    elif name == "LeftClick":
        cx,cy = action["coordinate"]
        page, client = click(page, client, cx, cy, time_wait=1000)

    elif name == "DoubleClick":
        cx,cy = action["coordinate"]
        page, client = doubleclick(page, client, cx, cy, time_wait=1000)

    elif name == "ClickMany":
        # ── Picture-grid CAPTCHA: click every square in one go ──────────────
        # See the comment on prompts.CLICK_MANY_ACTION_LINE. Each point is a real mouse
        # event (same path as click), just without waiting for the model in between;
        # CAPTCHA providers only accept trusted events, so no synthetic DOM events here.
        cfg = get_config()
        pts = action.get("points") or []
        if not cfg.enable_click_many:
            print("🚫 rejected click_many() (enable_click_many is off)")
            LAST_ACTION_NOTE = ("click_many() is NOT an available action and was REJECTED - nothing "
                                "was clicked. Use click() on one element at a time.")
        elif _CLICK_MANY_COUNT[0] >= MAX_CLICK_MANY_PER_TASK:
            print(f"🚫 click_many budget used up ({MAX_CLICK_MANY_PER_TASK})")
            LAST_ACTION_NOTE = (f"click_many() was REJECTED - already used {MAX_CLICK_MANY_PER_TASK} "
                                f"times on this task. If the verification puzzle keeps coming back, "
                                f"call finished(content='').")
        else:
            _CLICK_MANY_COUNT[0] += 1
            pts = pts[:MAX_CLICK_MANY_POINTS]
            for i, (px, py) in enumerate(pts):
                if i == len(pts) - 1 and len(pts) > 1:
                    # The last point is usually Verify: let the selected squares settle before submitting
                    page.wait_for_timeout(700)
                try:
                    if client is not None:
                        cdp_mouse_move(client, px, py, steps=8)
                except Exception:
                    pass
                page.mouse.click(px, py)
                page.wait_for_timeout(350)
            page.wait_for_timeout(1200)
            print(f"🔢 click_many {_CLICK_MANY_COUNT[0]}/{MAX_CLICK_MANY_PER_TASK}: clicked {len(pts)} point(s)")
            LAST_ACTION_NOTE = (
                f"click_many clicked {len(pts)} points in order. Look at the grid now: the tiles you "
                f"selected should show a mark. If some matching tiles are still unselected (or new "
                f"tiles appeared), select those with another click_many; otherwise click the "
                f"Verify / Skip / Next button.")

    elif name == "Hover":
        cx,cy = action["coordinate"]
        cdp_mouse_move(client, cx, cy, steps=20)

    elif name == "Select":
        cx, cy = action["coordinate"]
        opt = action.get("value", "")
        try:
            ok, note = select_option_at(page, cx, cy, opt)
            if ok:
                print(f"✅ select: chose '{opt}'")
                page.wait_for_timeout(800)   # let the change event re-render the page
            else:
                print(f"⚠️ select had no effect: {note[:110]}")
                LAST_ACTION_NOTE = note
        except Exception as e:
            print(f"⚠️ select failed (option='{opt}'): {e}")
            LAST_ACTION_NOTE = (f"select(option='{opt}') failed and did NOT run "
                                f"({str(e)[:160]}). Try a different approach.")

    elif name == "Fill":
        # ── Fill text into the input box at the coordinates ──────────────────
        # Operates only the page's own controls; goes nowhere new and builds no URL. One
        # fill is one input operation, like a person changing the value in a box once,
        # and counts as one step.
        #
        # What it fixes: in one development set **15 of 70 tasks typed exactly the same
        # content more than once** = the previous input did not take. type() sends keys to
        # whatever has focus, which depends on the previous coordinate click, and a slight
        # miss is invisible in the screenshot, so the model retried until the steps ran out.
        cfg = get_config()
        cx, cy = action["coordinate"]
        content = action.get("value", "")
        if not cfg.enable_fill:
            print("🚫 rejected fill() (enable_fill is off)")
            LAST_ACTION_NOTE = (
                "fill() is NOT an available action and was REJECTED - nothing was typed. "
                "Click the box first, then use type(content='...').")
        else:
            try:
                ok, note = fill_at(page, cx, cy, content, timeout_ms=cfg.fill_timeout_ms)
                print(("✏️ fill: " if ok else "⚠️ fill had no effect: ") + note[:110])
                LAST_ACTION_NOTE = note
            except Exception as e:
                print(f"⚠️ fill failed: {e}")
                LAST_ACTION_NOTE = (f"fill() failed and did NOT run ({str(e)[:160]}). "
                                    f"Try clicking the box and using type() instead.")

    elif name == "RightClick":
        cx, cy = action["coordinate"]
        cdp_mouse_move(client, cx, cy, steps=20)
        page.mouse.click(cx, cy, button="right")
        page.wait_for_timeout(3000)

    elif name == "HotKey":
        hot_key_value = action["value"]
        try:
            page, client = hotkey(page, client, hot_key_value)
        except UnsupportedHotkey as e:
            print(f"⛔ hotkey rejected (key='{hot_key_value}'): {e}")
            LAST_ACTION_NOTE = str(e)
        except Exception as e:
            print(f"⚠️ hotkey failed (key='{hot_key_value}'): {e}")
            LAST_ACTION_NOTE = (
                f"hotkey(key='{hot_key_value}') failed and did NOT run ({e}). "
                f"Try a different approach.")

    elif name == "Reload":
        # ── Reload the current page ───────────────────────────────────────────
        # Why this action exists: when a page never finished bootstrapping (JS bundle
        # missing, endless spinner, "JavaScript has been disabled"), the model had no
        # remedy. Browser shortcuts (F5 / ctrl+r) cannot reach Chrome's UI over CDP and
        # are rejected, and there is no goto. In one task the model wrote "let me try
        # refreshing the page" and then spent 18 steps on wait()/scroll() before giving
        # up. In one development set, 8 of 70 tasks kept complaining that the page had
        # not loaded, and 6 of them ended without an answer.
        #
        # reload goes nowhere new and builds no URL; it is the same as a person pressing
        # F5 on the current page.
        #
        # Budget: at most MAX_RELOADS_PER_TASK per task; beyond that it is rejected with a
        # reason, so it cannot become a new way to idle (asking does not work; the
        # executor has to enforce it).
        if _RELOAD_COUNT[0] >= MAX_RELOADS_PER_TASK:
            print(f"🚫 reload budget used up ({MAX_RELOADS_PER_TASK})")
            LAST_ACTION_NOTE = (
                f"reload() was REJECTED - you have already reloaded this page "
                f"{MAX_RELOADS_PER_TASK} times and the page did not change. Reloading again "
                f"will not help. Work with what is on screen, or if the data truly cannot be "
                f"reached by operating this page, call finished(content='').")
        else:
            _RELOAD_COUNT[0] += 1
            try:
                page.reload(wait_until="domcontentloaded", timeout=60000)
                settle_after_navigation(page)   # same wait as open_page's first paint
                print(f"🔄 reload {_RELOAD_COUNT[0]}/{MAX_RELOADS_PER_TASK}: {page.url[:80]}")
                LAST_ACTION_NOTE = (
                    "The page was reloaded. Anything you had opened, typed, filtered or "
                    "scrolled to on it is gone - read the screenshot and start from the "
                    "state that is actually there now.")
            except Exception as e:
                # A timeout does not mean nothing loaded: the page may be rendering even
                # without domcontentloaded. Report the reason; do not fail the task.
                print(f"⚠️ reload timed out / failed: {e}")
                LAST_ACTION_NOTE = (
                    f"reload() did not finish loading within the time limit ({str(e)[:120]}). "
                    f"The page may still be loading - look at the screenshot before deciding "
                    f"what to do next.")

    elif name == "ReadText":
        # ── Read text from the rendered page to **verify** a value read off the screenshot ──
        # A check, not a replacement for the screenshot, for two reasons:
        #   1. On sites with anti-scraping fonts the DOM holds garbage and only the
        #      screenshot shows the true value (see looks_obfuscated)
        #   2. The final screenshot is the evidence, so the answer must still be on screen
        #
        # It **does not move the view**. If the view changed after read_text, the agent
        # must return to the evidence before finished(); the prompt says so, and the
        # feedback here repeats it.
        cfg = get_config()
        near = (action.get("value") or "").strip()
        if not cfg.enable_read_text:
            print("🚫 rejected read_text() (enable_read_text is off)")
            LAST_ACTION_NOTE = (
                "read_text() is NOT an available action and was REJECTED. Read the value "
                "off the screenshot instead.")
        elif not near:
            LAST_ACTION_NOTE = (
                "read_text() needs words to locate, e.g. read_text(near='Total revenue'). "
                "The previous action did NOT run.")
        elif _READ_TEXT_COUNT[0] >= cfg.max_read_text_per_task:
            print(f"🚫 read_text budget used up ({cfg.max_read_text_per_task})")
            LAST_ACTION_NOTE = (
                f"read_text() was REJECTED - already used {cfg.max_read_text_per_task} times "
                f"on this task. Read the remaining values off the screenshot.")
        else:
            try:
                loc, in_frame, n = locate_text(page, near)
                if n == 0:
                    print(f"📖 read_text('{near[:40]}') not found")
                    LAST_ACTION_NOTE = (
                        f"read_text() found no element containing '{near[:60]}'. The wording may "
                        f"differ, or the value may be drawn in a chart/image rather than as text. "
                        f"Read it off the screenshot.")
                else:
                    _READ_TEXT_COUNT[0] += 1
                    el = loc.first
                    txt = el.inner_text() or ""
                    # The matched element is often just a label or a title; the value sits
                    # in a sibling on the same row. Walk up **in a loop**, not one level only.
                    # A measured parent chain:
                    #   L0 <a> = 4 chars, L1 <h2> = 4 chars, L2 <div> = 225 chars (with the value)
                    # Going up one level stops at L1 and returns the same 4 characters.
                    # At most 4 levels, stopping once the text is long enough: further up it
                    # grows to the whole list (L4 was already 3986 characters), and row
                    # context turns into burying the decision in unrelated text.
                    txt = txt.strip()
                    if len(txt) < 60:
                        cur = el
                        for _ in range(4):
                            try:
                                cur = cur.locator("xpath=..")
                                ptxt = (cur.inner_text() or "").strip()
                            except Exception:
                                break
                            if len(ptxt) > len(txt):
                                txt = ptxt
                            if len(txt) >= 60:
                                break
                    clean, n_bad = sanitize_page_text(txt.strip())
                    clipped = clean[:cfg.read_text_max_chars]
                    more = " …(truncated)" if len(clean) > cfg.read_text_max_chars else ""
                    print(f"📖 read_text {_READ_TEXT_COUNT[0]}/{cfg.max_read_text_per_task}: "
                          f"'{near[:30]}' matched {n} time(s), returned {len(clipped)} chars"
                          + (f", {n_bad} of them scrambled by an anti-scraping font" if n_bad else ""))
                    warn = ""
                    if n_bad:
                        warn = (f"\n⚠️ {n_bad} character(s) came back as '?' - this site scrambles "
                                f"its fonts, so those positions are NOT readable from the page "
                                f"source. Whatever the '?' hide, take that part from the "
                                f"SCREENSHOT, not from this text.")
                    LAST_ACTION_NOTE = (
                        f"read_text() returned the text from the page"
                        f"{' (from an embedded frame)' if in_frame else ''}:\n"
                        f"---\n{clipped}{more}\n---{warn}\n"
                        f"Compare it with what you read off the screenshot. The view did NOT "
                        f"move - make sure the answer is still visible on screen before "
                        f"calling finished().")
            except Exception as e:
                print(f"⚠️ read_text failed: {str(e)[:90]}")
                LAST_ACTION_NOTE = (
                    f"read_text() failed ({str(e)[:90]}). Read the value off the screenshot.")

    elif name == "FindText":
        # ── Find on page: scroll the target text into view ─────────────────────
        # One find_text is one scroll, i.e. one step.
        #
        # **Boundary, read this before changing the code**: only "locate + scroll into
        # view" is allowed. .inner_text() / .text_content() on the Locator would read the
        # text directly and **must never be used**: that would get data around the page's
        # interface, the same category as navigating by URL. The answer must still be
        # scrolled onto the screen and read by the model from the screenshot.
        #
        # Measured details (local long-page probe):
        #   - get_by_text() itself is lazy (0.0001 s, no browser round trip), so the whole
        #     thing is one scroll
        #   - several matches without .first raise a strict-mode violation; pick one explicitly
        #   - when the text does not exist it **waits out the full timeout** (30 s by
        #     default) before raising, so pass a short timeout
        #   - an element already in the viewport is not scrolled (if_needed checks itself)
        cfg = get_config()
        target = (action.get("value") or "").strip()
        if not cfg.enable_find_text:
            print("🚫 rejected find_text() (enable_find_text is off)")
            LAST_ACTION_NOTE = (
                "find_text() is NOT an available action and was REJECTED - the page did not "
                "change. Use scroll() to move around the page.")
        elif not target:
            LAST_ACTION_NOTE = (
                "find_text() needs the text to look for, e.g. find_text(text='Total revenue'). "
                "The previous action did NOT run.")
        elif _FIND_TEXT_COUNT[0] >= cfg.max_find_text_per_task:
            print(f"🚫 find_text budget used up ({cfg.max_find_text_per_task})")
            LAST_ACTION_NOTE = (
                f"find_text() was REJECTED - you have already used it "
                f"{cfg.max_find_text_per_task} times on this task. Work with what is on "
                f"screen, or scroll to look around.")
        else:
            try:
                loc, in_frame, n = locate_text(page, target)
                if n == 0:
                    # Not charged to the budget: nothing was operated, and the model should retry with other words.
                    print(f"🔍 find_text('{target[:40]}') not found")
                    LAST_ACTION_NOTE = (
                        f"find_text() found no text matching '{target[:80]}' on this page, and "
                        f"the view did not move. The wording may differ, the value may be inside "
                        f"a chart or an embedded frame, or you may be on the wrong page.")
                else:
                    _FIND_TEXT_COUNT[0] += 1
                    # Same text again = jump to the next match (see _FIND_LAST); new text starts at the first
                    k = (_FIND_LAST[1] + 1) % n if (_FIND_LAST[0] == target and n > 1) else 0
                    _FIND_LAST[0], _FIND_LAST[1] = target, k
                    el = loc.nth(k)
                    if in_frame:
                        el.evaluate(_SCROLL_INTO_VIEW_JS)     # see the comment on _SCROLL_INTO_VIEW_JS
                        page.wait_for_timeout(500)
                    else:
                        el.scroll_into_view_if_needed(timeout=cfg.find_text_timeout_ms)
                    print(f"🔍 find_text {_FIND_TEXT_COUNT[0]}/{cfg.max_find_text_per_task}: "
                          f"'{target[:40]}' matched {n} time(s), scrolled to #{k + 1}")
                    more = (f" There are {n} places on this page containing that text; the view "
                            f"moved to occurrence {k + 1} of {n}. Call find_text with the same text "
                            f"again to jump to the next occurrence.") if n > 1 else ""
                    LAST_ACTION_NOTE = (
                        f"The view scrolled to '{target[:80]}'"
                        f"{' (it is inside an embedded frame on this page)' if in_frame else ''}."
                        f"{more} Read the value off the "
                        f"screenshot - find_text only moves the view, it does not read anything "
                        f"for you.")
            except Exception as e:
                print(f"⚠️ find_text failed: {str(e)[:100]}")
                LAST_ACTION_NOTE = (
                    f"find_text() could not bring '{target[:60]}' into view "
                    f"({str(e)[:100]}). The text may be hidden or inside an embedded frame. "
                    f"Try scrolling, or look for it another way.")

    elif name == "Back":
        # ── Browser history back ──────────────────────────────────────────────
        # This branch covers two cases: normal execution with the switch on, and the
        # model writing back() from its training memory while the switch is off. The
        # latter must be **rejected**, not executed and judged afterwards (asking without
        # enforcing does not work).
        #
        # page.go_back() builds and guesses no address; it only returns to a page the
        # agent **reached by clicking in this session**, like pressing the browser's
        # back button.
        #
        # Why it is worth having: without back(), after following a list item into a
        # detail page the only way back was to redo the whole search; one task spent 12
        # of its 40 steps refilling the same address form.
        cfg = get_config()
        if not cfg.enable_back:
            print("🚫 rejected back() (enable_back is off)")
            LAST_ACTION_NOTE = (
                "back() is NOT an available action and was REJECTED - the page did not "
                "change. Move around only by operating what is on the screen.")
        elif _BACK_COUNT[0] >= cfg.max_backs_per_task:
            print(f"🚫 back budget used up ({cfg.max_backs_per_task})")
            LAST_ACTION_NOTE = (
                f"back() was REJECTED - you have already gone back "
                f"{cfg.max_backs_per_task} times on this task. Going back again will not "
                f"help. Work with what is on screen, or call finished() with what you have.")
        else:
            before = page.url
            try:
                resp = page.go_back(wait_until="domcontentloaded", timeout=60000)
                settle_after_navigation(page)   # same wait as open_page / reload
                if resp is None and page.url == before:
                    # Already at the start of the history. Not an error, but the model must be
                    # told, or it keeps repeating back() as something that "looks like work".
                    print("↩️ back(): no history to go back to; page unchanged")
                    LAST_ACTION_NOTE = (
                        "back() did nothing - there is no earlier page to return to. "
                        "You are at the first page of this task. Navigate forward by "
                        "operating what is on the screen.")
                else:
                    _BACK_COUNT[0] += 1
                    print(f"↩️ back {_BACK_COUNT[0]}/{cfg.max_backs_per_task}: "
                          f"{before[:60]} → {page.url[:60]}")
                    LAST_ACTION_NOTE = (
                        "You went back to the previous page. Anything you typed or "
                        "selected on the page you just left is gone - read the screenshot "
                        "and continue from the state that is actually there now.")
            except Exception as e:
                # A timeout does not mean no navigation: the page may be rendering even
                # without domcontentloaded. Report the reason; do not fail the task (as for reload).
                print(f"⚠️ back timed out / failed: {e}")
                LAST_ACTION_NOTE = (
                    f"back() did not finish within the time limit ({str(e)[:120]}). "
                    f"Look at the screenshot before deciding what to do next.")

    elif name == "Goto":
        # ── Navigation by URL is always rejected ──────────────────────────────
        # The agent must reach its answer by operating the site's own interface. Reaching
        # a page by URL, whether a search engine, a data API or a guessed URL with query
        # parameters, does not count, even if the number is right.
        #
        # goto was once added to the action space as a workaround and immediately led the
        # agent off course in most tasks: to search engines, third-party sites, internal
        # data endpoints and hand-built filter URLs.
        #
        # prompts.py no longer offers goto, so normally nothing reaches this branch. It
        # stays to catch the model writing goto from its training memory: **rejected**,
        # not executed. Asking alone does not work: both a static prompt constraint and
        # runtime feedback were tried and failed. "Back to the start page" is not a model
        # action; when needed, the runner calls open_page(start URL) itself.
        target = action.get("value", "")
        print(f"🚫 rejected goto(url='{target[:80]}')")
        LAST_ACTION_NOTE = (
            "goto() is NOT an available action and was REJECTED - the page did not change. "
            "You cannot navigate by URL at all. Move around only by operating what is on "
            "the screen: click links, buttons, tabs and menus, or type into the page's own "
            "search box. If the data cannot be reached that way, call finished(content='').")

    elif name == "Unknown":
        LAST_ACTION_NOTE = (
            "The previous action could not be parsed and did NOT run. "
            "Emit exactly one action from the Action Space.")

    elif "Scroll" in name:
        if name in ["ScrollUp", "ScrollDown", "ScrollLeft", "ScrollRight"]:
            if name == "ScrollUp":
                delta_x = 0
                delta_y = -(page_height - 100)
            elif name == "ScrollDown":
                delta_x = 0
                delta_y = page_height - 100
            elif name == "ScrollLeft":
                delta_x = -(page_width - 100)
                delta_y = 0
            elif name == "ScrollRight":
                delta_x = page_width - 100
                delta_y = 0
            
            if "coordinate" in action:
                x, y = action["coordinate"]
                page, client = scroll_menu(page, client, x, y, delta_x, delta_y)
            else:
                page, client = scroll(page, client, delta_x, delta_y)

        else:
            print(f"parse error: {name}")
            exit()

    elif name == "Drag":
        x1, y1, x2, y2 = action["bbox"]
        page, client = drag(page, client, x1, y1, x2, y2, steps=20)

    elif name == "Wait":
        page, client = wait(page, client, time=3000)
   
    # Detect newly opened pages and clean up the old ones
    # Pages added by the action
    current_pages = set(page.context.pages)
    new_pages = current_pages - old_pages
    print("len new pages:", len(list(new_pages)))

    final_page = page
    final_client = client

    # Return the new page if there is one, otherwise the original page
    if new_pages:
        new_page = list(new_pages)[0]
        new_page.set_viewport_size({"width": browser_window_size_width, "height": browser_window_size_height})
        new_client = page.context.new_cdp_session(new_page)
        print(f"New page detected; closing {len(old_pages)} old page(s)...")
        
        # closed_pages = old_pages
        # Keep the last old page so the browser does not close if the new page fails to load
        closed_pages = list(old_pages)[:-1]
        
        for p in closed_pages: 
            try:
                # run_before_unload=False forces the close and ignores "leave this page?" dialogs
                p.close(run_before_unload=False)
            except Exception as e:
                print(f"Failed to close an old page: {e}")

        final_page = new_page
        final_client = new_client

    wait_for_rendering_complete(final_page)

    # No-op detection: compare fingerprints once rendering has settled. A new tab obviously means a change.
    if not new_pages and not LAST_ACTION_NOTE:
        _desc = {
            "LeftClick": "The click", "DoubleClick": "The double-click",
            "RightClick": "The right-click", "Type": "Typing that text",
            "HotKey": "That key press", "Wait": "Waiting",
            "Goto": "That navigation", "Hover": "Hovering there",
        }.get(name, "The previous action")
        if "Scroll" in name:
            _desc = "That scroll"
        _fp_after = page_fingerprint(final_page)
        _frame_changed = False
        # Clicking into an iframe always makes that <iframe> the main frame's activeElement;
        # focus moved, but the page did not react. So for clicks on an iframe, `active` is
        # ignored when comparing main-frame fingerprints; otherwise a swallowed click would
        # count as effective because focus moved, and the fallback would never run
        # (confirmed in a real browser).
        _ignore = ("active",) if _frame_hit is not None else ()
        if _frame_hit is not None and not _fp_changed(_fp_before, _fp_after, _ignore, _text_hash):
            # Main frame unchanged != no effect: the change may be entirely inside the iframe (see child_frame_hit)
            _frame, _ox, _oy, _ffp_before = _frame_hit
            _frame_changed = _fp_changed(_ffp_before, frame_fingerprint(_frame), text_hash=_text_hash)
            if not _frame_changed and name in ("LeftClick", "Hover"):
                # The physical click/move was swallowed (at the bottom of the viewport, three
                # clicks in a row missed and no tooltip appeared) -> dispatch DOM events
                # (click or move) inside the child frame
                _cx, _cy = action["coordinate"]
                _js = js_click_in_frame if name == "LeftClick" else js_hover_in_frame
                _hit_el = _js(_frame, _cx - _ox, _cy - _oy)
                if _hit_el:
                    page.wait_for_timeout(1500 if name == "LeftClick" else 600)
                    _fp_after = page_fingerprint(final_page)
                    _frame_changed = (_fp_changed(_ffp_before, frame_fingerprint(_frame), text_hash=_text_hash)
                                      or _fp_changed(_fp_before, _fp_after, _ignore, _text_hash))
                    print(f"🖱️ iframe DOM {'click' if name == 'LeftClick' else 'hover'} fallback: "
                          f"<{_hit_el.get('tag')}> '{_hit_el.get('text', '')}' -> "
                          f"{'worked' if _frame_changed else 'still no change'}")
        if _frame_changed:
            _NO_CHANGE_STREAK[0] = 0        # a change in the child frame = the action worked, same as a main-frame change
        else:
            _note = _diff_note(_fp_before, _fp_after, _desc, _point, _ignore, _text_hash)
            if _note:
                print(f"⚠️ no effect: {_note[:120]}")
                LAST_ACTION_NOTE = _note

    return final_page, final_client

# Wait for rendering
def wait_for_rendering_complete(page):
    """Wait for the page to settle after an action: load / domcontentloaded -> page height
    stops changing -> body text is not empty. A timeout at any stage only warns."""
    try:
        # Wait for load
        page.wait_for_load_state("load", timeout=30000)
        #page.wait_for_load_state("networkidle", timeout=30000)
        page.wait_for_load_state("domcontentloaded", timeout=10000)

        # A quick state check; if this succeeds, the page is probably stable
        is_ready = page.evaluate("() => document.readyState")
        if is_ready != "complete":
            print(f"Warning: document.readyState is {is_ready}, not 'complete'")

        # Then wait for the page height to stop changing
        page.evaluate("""() => {
            return new Promise(resolve => {
                let lastHeight = document.body.scrollHeight;
                let checkCount = 0;
                const interval = setInterval(() => {
                    const currentHeight = document.body.scrollHeight;
                    if (currentHeight === lastHeight || checkCount > 10) {
                        clearInterval(interval);
                        resolve();
                    } else {
                        lastHeight = currentHeight;
                        checkCount++;
                    }
                }, 100);
            });
        }""")

        # Make sure there is actual content
        page.wait_for_function("""() => {
            const content = document.body.innerText.trim();
            return content.length > 0;
        }""", timeout=5000)

    except Exception as e:
        print(f"Warning: wait_for_rendering_complete encountered an error: {e}")
        # On error, wait for load once more
        try:
            page.wait_for_load_state("load", timeout=5000)
        except Exception:
            pass  # ignore errors from the second wait

# Get the elements marked by the page's highlight script (inherited, unused)
def get_vimc_elements(page):
    # Extract highlighted elements
    print("🛠️ running highlight_dom_sync...")
    # page.wait_for_timeout(1000)
    # result_handle = page.evaluate_handle("""() => window.__highlight_dom_sync__?.()""")
    # page.wait_for_timeout(10000)
    page.evaluate("""() => {
    window.__highlight_completed__ = false;
    Promise.resolve(window.__highlight_dom_sync__?.())
      .then(result => {
        window.__highlight_result__ = result;
        window.__highlight_completed__ = true;
      });
    }""")
    page.wait_for_timeout(2000)
    try:
        page.wait_for_function("""() => window.__highlight_completed__ === true""", timeout=30000)
        result_handle = page.evaluate_handle("""() => window.__highlight_result__""")
        element_handles = result_handle.get_properties()
        print(f"✨ highlighted elements: {len(element_handles)}")

        highlighted_elements = []
        for idx, handle in element_handles.items():
            try:
                obj = handle.json_value()
                if "attributes" in obj:
                    if "id" in obj["attributes"]:
                        obj["attributes"]["attributes_id"] = obj["attributes"]["id"]
                        del obj["attributes"]["id"]
                highlighted_elements.append(obj)
            except Exception as e:
                print(f"⚠️ failed to extract element {idx}: {e}")

        page.evaluate("""() => window.__clear_highlight_dom_sync__?.()""")
        page.wait_for_timeout(1000)
    except Exception as e:
        return []
    return highlighted_elements

def get_url(scribe_path):
    #scribe_path_raw = scribe_path.replace("document_details_parse_", "document_details_").replace("_addthink_fix_aug", "").replace("_addthink_aug", "").replace("_aug_list.json",".json")
    scribe_path_raw = scribe_path.replace("document_details_parse_", "document_details_").replace("_addthink_fix", "").replace("_addthink", "")
    print(scribe_path_raw)
    scribe_items = json.load(open(scribe_path_raw))
    title = scribe_items["title"]
    action_list = scribe_items["actions"]
    #print(action_list[0]["description"])
    if "Navigate to" not in action_list[0]["description"]:
        return None, title
    matchs = extract_between_keywords(action_list[0]["description"], "(", ")")
    if len(matchs) == 0:
        matchs = extract_between_keywords(action_list[0]["description"], "<", ">")
    url = matchs[0].split("?")[0]
    print(url, title)    
    return url, title

def check_url_accessible(cdp_url, url, timeout=5000):
    """Check whether a URL is reachable from the remote browser."""
    with sync_playwright() as p:
        headers = get_cdp_headers()
        browser = p.chromium.connect_over_cdp(cdp_url, **({"headers": headers} if headers else {}))
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        
        try:
            # response = page.goto(url, timeout=timeout, wait_until='domcontentloaded')
            response = page.goto(url, timeout=timeout, wait_until='commit')
            # Check the response status code
            if response and response.status < 400:
                browser.close()
                return True, response.status
            else:
                browser.close()
                return False, response.status if response else None

        except Exception as e:
            browser.close()
            return False, str(e)

# Request recorder
class RequestCollector:
    """Per-process recorder of network requests.

    The `capture.json` it produces is the record of how a task was solved, so the
    standard is **exactly this task's requests, no more and no less**.

    - **No more**: workers are long-lived, and one browser context runs a dozen tasks
      in a row. Listeners hang on *pages*, and pages survive across tasks, so requests
      from a page left over by the previous task would be recorded in this task's
      capture.json, where requests to another site (especially a search engine) would
      be impossible to explain. So each task starts with `clear()`, which unbinds every
      old listener, together with `open_page()`'s single-tab reset (see the comment
      there); only both layers together keep it clean.
    - **No less**: when a task opens a new tab, the main loop switches to it and calls
      `attach()` on the new page, so its requests are recorded too.
    - `attach()` is idempotent: switching back and forth between tabs attaches to the
      same page repeatedly, and without de-duplication each request would be recorded twice.
    """

    def __init__(self):
        self.requests = []
        self._bound = []  # [(page, handler)]; unbinding needs the original handler reference

    def attach(self, page):
        """Attach the recorder to a page. Idempotent for the same page."""
        if page is None or any(bound is page for bound, _ in self._bound):
            return
        handler = self.handle_request
        page.on("request", handler)
        self._bound.append((page, handler))

    def detach_all(self):
        """Unbind the listeners from all pages. Closed pages are tried too; failures are ignored."""
        for page, handler in self._bound:
            try:
                page.remove_listener("request", handler)
            except Exception:
                pass  # page closed / connection gone: the listener went with it
        self._bound = []

    def clear(self):
        """Start a new task: cut every listener from the previous task, then empty the buffer.

        The order matters: emptying first and unbinding second would let requests from
        old pages slip into the new task in between.
        """
        self.detach_all()
        self.requests = []

    def handle_request(self, request):
        """Record one request."""
        req_data = {
            "timestamp": time.time(),
            "url": request.url,
            "method": request.method,
            "headers": dict(request.headers),
            "resource_type": request.resource_type,
            "post_data": None,
            "post_text": None,
            "json_data": None
        }
        
        # Only XHR / fetch (data requests); static resources are skipped. What matters is whether data endpoints or search engines were used
        if request.resource_type not in ["xhr", "fetch"]:
            return
        
        raw = request.post_data_buffer
        if raw:
            req_data["raw_post_data"] = raw.hex()
            try:
                text = raw.decode("utf-8")
                req_data["post_data"] = text
                try:
                    req_data["json_data"] = json.loads(text)
                except:
                    pass
                if req_data["post_data"] is None:
                    return
            except UnicodeDecodeError:
                pass
        
        self.requests.append(req_data)
    
    def save_results(self, save_result_path="capture.json"):
        """Save the captured requests."""
        output = {
            "capture_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_requests": len(self.requests),
            "all_requests": self.requests
        }
        
        with open(save_result_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        
        print("Requests saved to capture.json")
        return len(self.requests)
