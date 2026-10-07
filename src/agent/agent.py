"""Model layer: screenshot -> messages -> model call -> parsed action -> page pixels.

Starts from the UI-TARS 1.5 reference implementation. What changed, in the order a
step goes through it:

  1. Screenshot size    VLM_MAX_PIXELS / UITARSAgent.predict()
     Screenshots larger than 1440x810 are downscaled before sending. Otherwise the
     API downscales them itself, the model answers in the downscaled image's
     coordinates, and the reference code read those as page coordinates, so every
     click landed at 3/4 of its target.
  2. Coordinate mapping parse_action_to_structure_output()
     The size of the image actually sent and the real viewport size are passed
     separately: page pixel = model coordinate x (viewport / sent).
  3. Reply cleanup      _truncate_to_first_action() / _SPECIAL_TOKEN
     Keep only the first Thought/Action pair (the model likes to invent the rest of
     the conversation, fake finished() included); strip self-generated chat-template
     tokens such as <|eot_id|>; drop trailing junk after the action (e.g. "<br>").
  4. Action feedback    _obs_content()
     When the executor reports that the last action had no effect or was rejected,
     that note is sent to the model in front of the screenshot.
  5. Model-call retries UITARSAgent.predict()
     Back off and retry on 403 / 429 / 5xx / timeouts; attempts and total time are
     set in config.json. A final failure returns ("client error", "FAIL") so the
     main loop ends the task quickly and still writes result.json.
  6. Usage logging      _log_usage(), off by default; for local diagnostics.

Unchanged: message assembly (system + user prompt + a sliding window of the last
history_n=5 screenshots), Thought/Action extraction, action-type detection.
"""
import ast
import base64
import io
import json
import logging
import logging.handlers
import math
import multiprocessing as mp
import os
import platform
import random
import re
import shutil
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from glob import glob
from io import BytesIO
from time import sleep
from typing import Dict, List

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from requests.exceptions import SSLError

import prompts
from runtime_config import get_config

import web_controller
import prompts

RE_BOX_TOKEN = re.compile(
    r"<\|box_start\|>\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"(?:\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\))?"
    r"\s*<\|box_end\|>",
    re.S,
)

RE_START_BOX_STR = re.compile(
    r"start_box\s*=\s*'?\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"(?:\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\))?"
    r"\s*'?",
    re.I,
)


FINISH_WORD = "Finish"
WAIT_WORD = "Wait"
CALL_USER = "CallUser"
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

# ── Size budget for screenshots sent to the model ─────────────────────────
# Measured with a calibration page: when a 1920x1080 screenshot is sent as is,
# the API first shrinks it to about 1440x810, and the model returns coordinates
# in that *shrunken* space. Reading them as page coordinates put every click at
# 75% of its target:
#
#   send 1920x1080 -> clicks off by 122 / 158 / 189 px, all three targets missed
#   send 1440x810  -> clicks off by 1 / 1 / 5 px, all three hit
#
# Before this fix most tasks never finished: the agent kept clicking empty space.
# The value is the largest size the API passes through without rescaling, so
# coordinates map 1:1 onto the image we sent. The model's own pointing accuracy
# (0-3 px) was never the problem.
VLM_MAX_PIXELS = int(os.environ.get("WEBFOVEA_VLM_MAX_PIXELS", 1440 * 810))
MAX_RATIO = 200

def extract_between_keywords(text, start_key, end_key):
    pattern = re.compile(f'{re.escape(start_key)}(.*?){re.escape(end_key)}', re.DOTALL)
    matches = pattern.findall(text)
    return [match.strip() for match in matches]

def parse_action(action_str):
    """Parse action string using ast (aligned with uitars15_v1)."""
    try:
        node = ast.parse(action_str, mode='eval')
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")
        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # older Python versions
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value
        return {'function': func_name, 'args': kwargs}
    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None

def escape_single_quotes(text):
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def add_box_token(input_string):
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            updated_action = action
            for coord_type, x, y in coordinates:
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):
    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")
    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    ]
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text
                if '"' not in node.text
                else '"{:}"'.format(node.text.replace('"', '""'))
            )
        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith(
            "EditWrapper"
        ) and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (
                node_text
                if '"' not in node_text
                else '"{:}"'.format(node_text.replace('"', '""'))
            )
        else:
            text = '""'
        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                (
                    node.get("{{{:}}}class".format(_attributes_ns), "")
                    if platform == "ubuntu"
                    else node.get("{{{:}}}class".format(class_ns_windows), "")
                ),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get("{{{:}}}screencoord".format(_component_ns), ""),
                node.get("{{{:}}}size".format(_component_ns), ""),
            )
        )
    return "\n".join(linearized_accessibility_tree)

def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    return linearized_accessibility_tree

def get_start_index(base_directory):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory, exist_ok=True)
        return 0
    folders = [d for d in os.listdir(base_directory)
               if os.path.isdir(os.path.join(base_directory, d))]
    index_folders = []
    for folder in folders:
        match = re.match(r'^(\d+)_', folder)
        if match:
            index = int(match.group(1))
            index_folders.append((index, folder))
    if not index_folders:
        return 0
    max_index, max_folder = max(index_folders, key=lambda x: x[0])
    folder_path = os.path.join(base_directory, max_folder)
    result_path = os.path.join(folder_path, 'result.json')
    if not os.path.exists(result_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Removed folder without result.json: {folder_path}")
        except OSError as e:
            print(f"Failed to remove folder: {e}")
        return max_index
    return max_index + 1

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(text, factor, origin_resized_height, origin_resized_width, model_type, max_pixels=16384*28*28, min_pixels=100*28*28, screen_width=None, screen_height=None):
    """Parse model output: extract thought, split actions, remap coordinates.

    Returns list of dicts:
        - text: full model output
        - thought: extracted thought string
        - raw_action: original action string as model output
        - action: action string with coordinates remapped to screen pixels
        - action_type: ast function name for terminal detection (finished/wait/call_user)
    """
    text = text.strip()

    # Calculate smart_resize dimensions (what model actually sees)
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height, origin_resized_width,
            factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)

    # Model coordinates -> real page pixels.
    # origin_resized_* is the size of the image *actually sent*; screen_* is the size
    # of the *real page (viewport)*. They differ whenever the screenshot was
    # downscaled, so they must be passed separately. The reference code treated them
    # as one value and divided by smart_resize (written for UI-TARS's 28-px patch
    # grid), which for Claude amounted to no mapping at all: every click was off by
    # a factor of 1.333. See the VLM_MAX_PIXELS comment at the top of this file.
    #
    # No factor is hard-coded on purpose: the viewport size depends on the browser
    # window (web_controller.open_page), so it can differ between machines and the
    # factor has to be computed every time.
    if screen_width is None:
        screen_width = origin_resized_width
    if screen_height is None:
        screen_height = origin_resized_height
    sent_width, sent_height = origin_resized_width, origin_resized_height

    # Coordinate remapping helper
    def remap_xy(x, y):
        """Convert model coordinates to screen pixels."""
        rx = round(float(x) / sent_width * screen_width)
        ry = round(float(y) / sent_height * screen_height)
        return rx, ry

    def remap_action_str(raw_str):
        """Replace coordinates in action string with remapped screen pixel values."""
        def replace_box_token(match):
            coord_type = match.group(1)
            x, y = match.group(2), match.group(3)
            rx, ry = remap_xy(x, y)
            return f"{coord_type}='<|box_start|>({rx},{ry})<|box_end|>'"

        def replace_simple(match):
            coord_type = match.group(1)
            x, y = match.group(2), match.group(3)
            rx, ry = remap_xy(x, y)
            return f"{coord_type}='({rx},{ry})'"

        # Handle <|box_start|>(x,y)<|box_end|> format first
        result = re.sub(
            r"(start_box|end_box)='<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>'",
            replace_box_token, raw_str)
        # Then handle simple (x,y) format
        result = re.sub(
            r"(start_box|end_box)='\((\d+),\s*(\d+)\)'",
            replace_simple, result)

        # click_many(points='(x,y);(x,y);...'): remap every pair in the string
        def replace_points(match):
            def rp(m):
                rx, ry = remap_xy(m.group(1), m.group(2))
                return f"({rx},{ry})"
            return "points='" + re.sub(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", rp, match.group(1)) + "'"
        result = re.sub(r"points='([^']*)'", replace_points, result)
        return result

    # 1. Extract thought
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"

    thought = None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()

    assert "Action:" in text
    raw_action_str = text.split("Action:")[-1].strip()
    # Drop anything after the action. In one run the model followed every action with
    # "\n<br>"; finished(content='...')\n<br> then failed ast parsing, the fallback regex
    # requires a trailing ")", and a correct answer came out as an empty agent_answer.
    # Cutting at the last ")" leaves parentheses inside the answer intact.
    m = re.match(r"(?s)^(\w+\(.*\))", raw_action_str)
    if m:
        raw_action_str = m.group(1)

    if "type(content" in raw_action_str:
        def escape_quotes(match):
            return match.group(1)
        pattern = r"type\(content='(.*?)'\)"
        content = re.sub(pattern, escape_quotes, raw_action_str)
        type_str = escape_single_quotes(content)
        raw_action_str = "type(content='" + type_str + "')"

    raw_action_str = raw_action_str.strip()
    remapped_action_str = remap_action_str(raw_action_str)

    result = {
        "text": text,
        "thought": thought,
        "raw_action": raw_action_str,       # action string as the model wrote it
        "action": remapped_action_str,       # same, with coordinates mapped to page pixels
    }

    return result


def _log_usage(agent, response, started_at, attempt):
    """Log the token usage of one model call to the worker log. **Off by default.**

    It must never affect the run itself: the whole body is wrapped in try and every
    exception is swallowed. A diagnostic that breaks the real work is the worst
    possible trade.

    Why log it at all: `response.usage` used to be discarded, so there was no way to
    tell whether prompt caching kicked in. Reading the code suggests it does not
    (the history_n=5 image window shifts every step, rewriting the middle of the
    message list, and the OpenAI-compatible endpoint receives no cache_control).
    At a few hundred dollars per 100-task run, that is worth confirming with data.

    Uses the logger rather than a separate file: no plumbing changes, and the
    worker log is already split per task.
    """
    try:
        if not get_config().log_usage:
            return
        u = getattr(response, "usage", None)
        if u is None:
            return
        try:
            data = u.model_dump()          # pydantic: includes the cache-related extension fields
        except Exception:
            data = {k: getattr(u, k, None)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        data["elapsed_s"] = round(time.time() - started_at, 2)
        data["attempt"] = attempt + 1
        data["model"] = getattr(response, "model", None)
        data["finish_reason"] = response.choices[0].finish_reason if response.choices else None
        data["n_messages"] = len(agent.history_responses) * 2 + 3   # rough; for watching context growth
        if agent.logger:
            agent.logger.info("[usage] " + json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


#: The model occasionally emits **chat-template end tokens** (`<|eot_id|>` 471 times,
#: `<|end_of_turn|>` 80 times in our logs), most likely imitating the `<|box_start|>`
#: notation in the prompt. They have to be stripped here because they do damage in
#: two ways, and the second is the fatal one:
#:   1. Left inside `type(content=...)`, the token is **typed into the input box as
#:      text**. One episode typed `2022 June<|end_of_turn|>` into a filter box and the
#:      site kept answering "No Results". The model even diagnosed this in its own
#:      thoughts ("the type command keeps appending ... to the text") but could not
#:      get around it.
#:   2. The reply goes into `history_responses` unchanged, so on the next step the
#:      model sees itself using the token and keeps doing it. It hit 15 of 305
#:      episodes (4.9%); every one started at step 0, carried the token on every step
#:      after that, and scored zero, and 14 of them never called finished() at all.
#: Only the box tokens used for coordinates are let through; everything else is cut.
#: A regex rather than a fixed list: adding tokens one at a time (`<|im_end|>` was
#: handled that way) always left others uncovered.
_SPECIAL_TOKEN = re.compile(r"<\|(?!box_start\|>|box_end\|>)[A-Za-z0-9_]+\|>")


def _truncate_to_first_action(text: str) -> str:
    """Keep only the first Thought/Action pair and drop the turns the model made up.

    The model often invents a whole interaction in a single reply:
        Thought: Let me click the "Confirmed Planets" link first.
        Action: click(...)
        <environment>The screenshot has changed after the action.</environment>
        Thought: ...
        Action: scroll(...)
        ...
        Action: finished(content='14')      <- made up without ever seeing the page
    parse_action searched the *whole* text for an action, and any `finished(` ended
    the task, so the agent could submit a fabricated answer at step 0 (one task was
    answered 14 when the truth was 34). That is far worse than giving no answer.

    The same place also strips self-generated chat-template tokens; see _SPECIAL_TOKEN.
    """
    if not text:
        return text
    # Markers the model uses to fake "the environment sent me a new screenshot"
    for marker in ("<environment>", "<image>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    # Self-generated special tokens (`<|eot_id|>` etc.): cut at the first one, along with everything after it
    m = _SPECIAL_TOKEN.search(text)
    if m:
        text = text[:m.start()]
    # No marker: fall back to cutting at the first Thought that follows the first Action
    first_action = text.find("Action:")
    if first_action != -1:
        nxt = text.find("Thought:", first_action)
        if nxt != -1:
            text = text[:nxt]
    return text.strip()


def _obs_content(encoded_image: str, note: str = None):
    """Message content for the current observation. If the executor reported that the
    last action had no effect (web_controller.LAST_ACTION_NOTE), a text note goes in
    front of the screenshot. Without it a silently failed action is invisible to the
    model: it only sees an unchanged screenshot and repeats the action until the step
    budget runs out."""
    content = []
    if note:
        content.append({
            "type": "text",
            "text": f"[Action feedback] {note}",
        })
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
    })
    return content


class UITARSAgent:
    """UI-TARS 1.5 Agent - VLM-based GUI agent (aligned with uitars15_v1, no runtime_conf)."""

    def __init__(
        self,
        model: str = "uitars",
        api_url: str = "http://localhost:8001/v1",
        api_key: str = None,
        model_type: str = "qwen25vl",
        platform: str = "ubuntu",
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        max_trajectory_length: int = 50,
        a11y_tree_max_tokens: int = 10000,
        temperature: float = 0.7,
        top_k: int = -1,
        top_p: float = 1.0,
        max_tokens: int = 8192,
        history_n: int = 5,
        max_pixels: int = VLM_MAX_PIXELS,   # measured limit, see the comment at the top of this file
        min_pixels: int = 100 * 28 * 28,
        language: str = "Chinese",
        infer_mode: str = "qwen2vl_user",
        prompt_style: str = "qwen2vl_user",
        input_swap: bool = True,
        callusr_tolerance: int = 0,
    ):
        if api_key is None:
            api_key = os.getenv("VLM_API_KEY", "EMPTY")
        # max_retries=0 turns off the SDK's own retries; stacked on top of the backoff in
        # predict(), they made the time spent on a failing call unpredictable.
        self.vlm = OpenAI(base_url=api_url, api_key=api_key, timeout=120.0, max_retries=0)
        self.model = model
        self.model_type = model_type
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.history_n = history_n
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.language = language
        self.infer_mode = infer_mode
        self.prompt_style = prompt_style
        self.input_swap = input_swap
        self.callusr_tolerance = callusr_tolerance
        self.action_parse_res_factor = 1000

        # Select action space based on infer_mode (from prompts module)
        self.prompt_action_space = prompts.UITARS_ACTION_SPACE
        if self.infer_mode == "qwen2vl_user":
            self.prompt_action_space = prompts.UITARS_CALL_USR_ACTION_SPACE
        elif self.infer_mode == "qwen25vl_normal":
            # Whether each added action (back / find_text / read_text / fill / click_many /
            # the answer-provenance rule) appears in the action space is set by switches in
            # config.json; see prompts.normal_action_space(). A function rather than a
            # constant, so "switched off means the model never sees it" can be tested.
            _cfg = get_config()
            self.prompt_action_space = prompts.normal_action_space(
                _cfg.enable_back, _cfg.enable_find_text, _cfg.enable_read_text,
                _cfg.enable_fill, _cfg.enable_provenance_rule, _cfg.enable_click_many)

        # Select prompt template based on prompt_style (from prompts module)
        self.prompt_template = prompts.UITARS_USR_PROMPT_THOUGHT
        if self.prompt_style in ("qwen2vl_user", "qwen25vl_normal"):
            self.prompt_template = prompts.UITARS_USR_PROMPT_THOUGHT
        elif self.prompt_style == "qwen2vl_no_thought":
            self.prompt_template = prompts.UITARS_USR_PROMPT_NOTHOUGHT

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.urls = []
        self.logger = logging.getLogger(__name__)
        self.cur_callusr_count = 0

    def reset(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.urls = []
        self.cur_callusr_count = 0

    def predict(
        self, instruction: str, obs: Dict, last_action_after_obs: Dict = None
    ) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        # Flow: record this step's screenshot -> build messages (prompt + recent screenshots
        # + past replies + current screenshot) -> call the model (with retries) -> clean the
        # reply -> parse the action and map coordinates -> record history -> check for the end

        # Append trajectory
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(
            self.thoughts
        ), "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length :]
                _actions = self.actions[-self.max_trajectory_length :]
                _thoughts = self.thoughts[-self.max_trajectory_length :]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts


        self.history_images.append(obs["screenshot"])

        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = obs["screenshot"]
            try:
                linearized_accessibility_tree = (
                    linearize_accessibility_tree(
                        accessibility_tree=obs["accessibility_tree"],
                        platform=self.platform,
                    )
                    if self.observation_type == "screenshot_a11y_tree"
                    else None
                )
            except:
                linearized_accessibility_tree = None

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append(
                    {
                        "screenshot": base64_image,
                        "accessibility_tree": linearized_accessibility_tree,
                    }
                )
            else:
                self.observations.append(
                    {"screenshot": base64_image, "accessibility_tree": None}
                )

        else:
            raise ValueError(
                "Invalid observation_type type: " + self.observation_type
            )
        
        if self.infer_mode == "qwen2vl_user" or self.infer_mode == "qwen25vl_normal":
            user_prompt = self.prompt_template.format(
                instruction=instruction,
                action_space=self.prompt_action_space,
                language=self.language
            )
        elif self.infer_mode == "qwen2vl_no_thought":
            user_prompt = self.prompt_template.format(
                instruction=instruction
            )

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]

        messages, images = [], []
        if isinstance(self.history_images, bytes):
            self.history_images = [self.history_images]
        elif isinstance(self.history_images, np.ndarray):
            self.history_images = list(self.history_images)
        elif isinstance(self.history_images, list):
            pass
        else:
            raise TypeError(f"Unidentified images type: {type(self.history_images)}")

        for turn, image in enumerate(self.history_images):
            if len(images) >= self.history_n:
                break
            try:
                image = Image.open(BytesIO(image))
            except Exception as e:
                raise RuntimeError(f"Error opening image: {e}")

            # The size before downscaling is the real page (viewport) size, the space
            # coordinates must be mapped back to. The loop runs oldest to newest, so the
            # last image is the current step's screenshot.
            page_width, page_height = image.width, image.height

            if image.width * image.height > self.max_pixels:
                resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                image = image.resize((width, height))
            if image.width * image.height < self.min_pixels:
                resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
                width, height = math.ceil(image.width * resize_factor), math.ceil(image.height * resize_factor)
                image = image.resize((width, height))

            if image.mode != "RGB":
                image = image.convert("RGB")

            images.append(image)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
        
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                # send at most history_n images to the model
                if history_idx + self.history_n > len(self.history_responses):

                    cur_image = images[image_num]
                    encoded_string = pil_to_base64(cur_image)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                    })
                    image_num += 1
                    
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": add_box_token(history_response)}]
                })

            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": _obs_content(encoded_string, obs.get("note"))
            })
            image_num += 1
        
        else:
            cur_image = images[image_num]
            encoded_string = pil_to_base64(cur_image)
            messages.append({
                "role": "user",
                "content": _obs_content(encoded_string, obs.get("note"))
            })
            image_num += 1

        # Size of the image actually sent (possibly downscaled by the max_pixels budget)
        origin_resized_height = images[-1].height
        origin_resized_width = images[-1].width
        if origin_resized_width != page_width:
            print(f"[coord] screenshot {page_width}x{page_height} -> sent "
                  f"{origin_resized_width}x{origin_resized_height}, "
                  f"coordinate scale {page_width / origin_resized_width:.4f}")
        temperature = self.temperature
        top_k = self.top_k
        prediction = None
        # Attempts and total time are configurable; see runtime_config.model_call_max_attempts
        # (with a fixed 3 attempts, three 403s in a row at step 0 ended a task before it began).
        _cfg = get_config()
        max_retries = max(1, _cfg.model_call_max_attempts)
        _retry_started = time.time()
        for attempt in range(max_retries):
            try:
                _call_started = time.time()
                response = self.vlm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    frequency_penalty=1,
                    max_tokens=self.max_tokens,
                    temperature=temperature
                )
                _log_usage(self, response, _call_started, attempt)
                print("*" * 20)
                print("Response:")
                print(response.choices[0].message.content)
                print("*" * 20)
                prediction = _truncate_to_first_action(
                    response.choices[0].message.content.strip())
                break
            except Exception as e:
                # 429 rate limits, 5xx and network timeouts are worth a retry; other 4xx
                # errors (bad parameters) are not. 403 is the exception: we saw sporadic
                # `403 Request not allowed` responses mid-task, each of which used to kill
                # the task on the spot (2 of 10 tasks in one run). They are transient
                # policy refusals that usually pass on retry; treating them like a 400 for
                # a malformed request was far too harsh.
                status_code = getattr(e, "status_code", None)
                retryable = (status_code is None or status_code == 429
                             or status_code == 403 or status_code >= 500)
                self.logger.warning(
                    f"Model call failed (attempt {attempt + 1}/{max_retries}, status={status_code}): {e}")
                if (not retryable or attempt == max_retries - 1
                        or time.time() - _retry_started > _cfg.model_call_retry_budget_s):
                    break
                # 403/429 usually clear within seconds; cap the wait at 30 s per attempt
                time.sleep(min(2 ** attempt * 2, 30))

        if prediction is None:
            # This used to return ["DONE"] (a list), which main.py's `condition in ["DONE"]`
            # never matched, so the task idled until the step limit. A failed model call fails
            # the task; end it quickly and leave the time to other tasks.
            return "client error", "FAIL"

        # Parse model output → browser-ready action dicts (one step)
        try:
            parsed_response = parse_action_to_structure_output(
                prediction,
                self.action_parse_res_factor,
                origin_resized_height,
                origin_resized_width,
                self.model_type,
                self.max_pixels,
                self.min_pixels,
                screen_width=page_width,     # real page size, not the size sent
                screen_height=page_height,
            )
        except Exception as e:
            print(f"Parsing action error: {prediction}, with error:\n{e}")
            # Same as above: a list slips past main.py's check. Parsing failed, so fail fast
            return f"Parsing action error: {prediction}, with error:\n{e}", "FAIL"

        # Record history
        thought_str = parsed_response["thought"]
        action_str = parsed_response["action"]
        self.thoughts.append(thought_str)
        self.actions.append(action_str)
        self.history_responses.append(prediction)

        # Collect browser actions, check for terminal conditions
        action_type = web_controller.parse_action_type(action_str)
        if action_type == FINISH_WORD:
            return prediction, "DONE"
        elif action_type == WAIT_WORD:
            return prediction, "WAIT"
        elif action_type == CALL_USER:
            if self.callusr_tolerance > self.cur_callusr_count:
                self.cur_callusr_count += 1
                return prediction, "WAIT"
            else:
                return prediction, "FAIL"

        if len(self.history_responses) >= self.max_trajectory_length:
            return prediction, "FAIL"

        return prediction, "CONTINUE"

    def get_thoughts(self):
        return self.thoughts

    def get_actions(self):
        return self.actions

    def get_history_responses(self):
        return self.history_responses

    def get_history_images(self):
        return self.history_images

    def get_urls(self):
        return self.urls
