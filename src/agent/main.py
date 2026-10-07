"""Task runner (called by scripts/run.sh): start worker processes; each worker
claims tasks, runs them and writes the results.

Starts from the reference implementation's main.py. Changes, in the order of a
task's lifecycle:

  Dispatch  static np.array_split sharding -> TaskDispatcher shared queue
            (task_dispatch.py). A free worker claims the next task; a crashed worker
            process is restarted by the parent (supervise_workers).
            Each task may be claimed only once (max_task_attempts=1): a task that
            has started is never re-run from scratch.
  Open      browser initialization is retried worker_init_retries times. After
            max_consecutive_page_failures tasks in a row whose page never opened,
            the browser is considered dead and the worker exits, leaving the tasks
            to others.
  Run       added to the per-task loop (while effect_step < total_steps in run_worker):
            - wall-clock budget per task (task_budget_s), so one stuck task cannot
              eat the time of the others
            - screenshot timeout: bring_to_front() and try once more before giving up
            - no action from the model at step 0: end as FAIL normally, not through
              the exception fallback
            - an action that raises no longer ends the task; the error goes back to
              the model and the loop continues
            - lost browser: with max_reconnect_per_task=0 (the shipped config) the task
              ends as FAIL_BROWSER_LOST; only with >0 is the start page reopened
            - scroll breaker: idle only if "10 scrolls in a row AND the page
              fingerprint did not change", so reading a long table screen by screen
              is not cut off
            - step accounting: click_many counts one step per square clicked
              (step_cost)
  Save      every exit path writes result.json (normal / breaker / timeout /
            exception fallback); agent_answer is taken from the last
            finished(content=...) (extract_agent_answer). capture.json is written in
            exactly one place, the finally block, and the capture_armed gate keeps
            the previous task's requests out of it.
  Observe   print_boot_banner() prints the Python/Playwright versions and the full
            configuration at the very top of stdout.

The external interface is unchanged: run.sh's positional arguments, number of CDP
URLs = number of workers, output directory layout.
"""
import argparse
import ast
import platform
import re
import gc
import json
import logging
import logging.handlers
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback
import fcntl
from datetime import datetime
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import web_controller
from agent import UITARSAgent
from runtime_config import get_config
from task_dispatch import TaskDispatcher, supervise_workers

# ── Runtime parameters ─────────────────────────────────────────────────────
# Step limit, wall-clock budget, reconnect limit, reference_length fallback, settle
# window: the reasoning behind every value and the priority order (env var >
# config.json["agent"] > default) are in runtime_config.py. get_config() loads once
# per process and is shared with web_controller.
CFG = get_config()
TASK_BUDGET_S = CFG.task_budget_s
MAX_RECONNECT_PER_TASK = CFG.max_reconnect_per_task

_DISCONNECT_HINTS = (
    "has been closed", "target closed", "target crashed", "disconnected",
    "connection closed", "browser has been closed", "websocket",
)


def looks_disconnected(err_msg):
    m = (err_msg or "").lower()
    return any(h in m for h in _DISCONNECT_HINTS)


def extract_agent_answer(actions):
    """Take the answer from the finished(content=...) action for result.json's agent_answer.

    Scoring reads only this field; empty means zero. Search backwards for the last
    finished action: parse with ast first, fall back to a regex (ast raises when the
    answer contains a single quote).
    """
    for act in reversed(actions or []):
        a = (act or "").strip()
        if not (a.startswith("finished(") or a.startswith("finish(")):
            continue
        try:
            node = ast.parse(a, mode="eval").body
            for kw in getattr(node, "keywords", []):
                if kw.arg == "content" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value).strip()
        except Exception:
            pass
        m = re.search(r"""content\s*=\s*(['"])(.*)\1\s*\)\s*$""", a, re.DOTALL)
        if m:
            return m.group(2).strip()
        # With trailing junk (e.g. "\n<br>") neither of the above matches; fall back to
        # requiring only "quote + )". The greedy match runs to the last occurrence, so
        # quotes and parentheses inside the answer do not cut it short.
        m = re.search(r"""content\s*=\s*(['"])(.*)\1\s*\)""", a, re.DOTALL)
        if m:
            return m.group(2).strip()
    return ""


def save_debug_image(image_path, bbox, point, markdown, action, save_dir):
    # action/markdown may arrive as lists
    if isinstance(action, list): action = str(action)
    if isinstance(markdown, list): markdown = str(markdown)
    show_image = cv2.imread(image_path)
    if show_image is None:
        print(f"[save_debug_image] failed to read image: {image_path}")
        return

    if bbox is not None:
        try:
            left, top, right, bottom = bbox
            print(bbox)
            left, top, right, bottom = map(int, [left, top, right, bottom])
            # Note: the trailing 0 is dropped; only the first 5 arguments are passed
            cv2.rectangle(show_image, (left, top), (right, bottom), (0, 0, 255), 4)
        except Exception as e:
            print(f"[save_debug_image] cannot draw bbox={bbox}, err={e}")

    if point is not None:
        try:
            cx, cy = point
            cx, cy = int(cx), int(cy)
            cv2.circle(show_image, (cx, cy), 4, (0, 0, 255), -1)
        except Exception as e:
            print(f"[save_debug_image] cannot draw point={point}, err={e}")
    pil_img = Image.fromarray(cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    microhei_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    zenhei_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    font_size = 24
    font = None
    system = platform.system()
    try:
        if system == "Windows":
            possible_fonts = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "C:/Windows/Fonts/msyhbd.ttc",
            ]
        else:
            possible_fonts = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
                "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc"
            ]
        for path in possible_fonts:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
    except Exception as e:
        pass
    if font is None:
        font = ImageFont.load_default()
    text_x = 10
    text_y = 10
    line_spacing = 35
    markdown_position = (text_x, text_y)
    markdown_bbox = draw.textbbox(markdown_position, markdown, font=font)
    padding = 5
    draw.rectangle(
        [markdown_bbox[0] - padding, markdown_bbox[1] - padding,
         markdown_bbox[2] + padding, markdown_bbox[3] + padding],
        fill=(255, 255, 255, 200)
    )

    draw.text(markdown_position, markdown, font=font, fill=(0, 128, 0))
    action_position = (text_x, text_y + line_spacing)
    action_bbox = draw.textbbox(action_position, action, font=font)
    draw.rectangle(
        [action_bbox[0] - padding, action_bbox[1] - padding,
         action_bbox[2] + padding, action_bbox[3] + padding],
        fill=(255, 255, 255, 200)
    )
    draw.text(action_position, action, font=font, fill=(0, 0, 255))
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    imageName = image_path.split("/")[-1]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sava_path = os.path.join(save_dir, imageName)
    cv2.imwrite(sava_path, result_img)
     
def safe_create_directory(base_dir, result_file="result.json"):
    # .lock files go into a locks/ subdirectory so they do not mix with task directories
    parent_dir = os.path.dirname(base_dir)
    lock_dir = os.path.join(parent_dir, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_name = os.path.basename(base_dir) + ".lock"
    lock_file = os.path.join(lock_dir, lock_name)
    try:
        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.path.exists(os.path.join(base_dir, result_file)):
                return False
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(f"{base_dir}/trajectory", exist_ok=True)
            os.makedirs(f"{base_dir}/trajectory_visual", exist_ok=True)
            return True
    except BlockingIOError:
        return False
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except:
            pass

def safe_remove_task_dir(task_dir):
    parent_dir = os.path.dirname(task_dir)
    lock_dir = os.path.join(parent_dir, "locks")
    lock_name = os.path.basename(task_dir) + ".lock"
    lock_file = os.path.join(lock_dir, lock_name)
    if os.path.exists(task_dir):
        try:
            shutil.rmtree(task_dir)
        except Exception as e:
            print(f"Error removing task directory {task_dir}: {e}")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except Exception as e:
            print(f"Warning: Failed to remove lock file {lock_file}: {e}")

def safe_save_results(base_dir, result_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            temp_result_path = f"{base_dir}/.result.json.tmp"
            with open(temp_result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
            with open(temp_result_path, 'r', encoding='utf-8') as f:
                json.load(f)
            os.replace(temp_result_path, f"{base_dir}/result.json")
            return True
        except Exception as e:
            if os.path.exists(temp_result_path):
                try:
                    os.remove(temp_result_path)
                except:
                    pass
            if attempt == max_retries - 1:
                print(f"Failed to save results after {max_retries} attempts: {e}")
                return False
            time.sleep(0.1 * (2 ** attempt))
    return False


def save_capture_json(request_collector, base_dir, armed, logger=None):
    """Write the network requests captured for this task to `capture.json`. Returns
    whether the file was actually written.

    `capture.json` is the record of how the agent got its answer: it shows that the
    agent operated the site itself and did not go through search engines or data
    APIs. Without it, a task's trajectory cannot be checked.

    `armed` is not optional; it is **the gate against cross-task contamination**.
    `request_collector` is reused across tasks and still holds the previous task's
    requests until `clear()` is called. The page-open failure path never reaches
    `clear()`, so writing the file there would put the previous task's requests into
    this task's directory, a bug that had already been fixed once. **No file is
    better than a wrong one**: the first is missing evidence, the second is false
    evidence.

    Never raises: it is called from `finally`, in exactly the situations where
    something has already gone wrong, and raising from there would replace the
    original exception and take the whole task loop down with it.
    """
    if not armed:
        return False
    path = f"{base_dir}/capture.json"
    try:
        n = request_collector.save_results(path)
        if logger:
            logger.info(f"saved {n} requests to {path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"failed to write capture.json (this task's network record will be missing): {e}")
        return False


def safe_write_json(filepath, data, max_retries=3):
    dir_path = os.path.dirname(filepath)
    os.makedirs(dir_path, exist_ok=True)
    
    # Remember the temp file name so it can be cleaned up
    tmp_name = None
    
    for attempt in range(max_retries):
        try:
            # --- Plan A: atomic write (the standard approach) ---
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=dir_path,
                delete=False,
                suffix='.tmp',
                encoding='utf-8'
            ) as tmp_file:
                json.dump(data, tmp_file, indent=4, ensure_ascii=False)
                tmp_file.flush()
                os.fsync(tmp_file.fileno()) # force it to disk
                tmp_name = tmp_file.name
            
            # Fix permissions: tempfile defaults to 600; mounted volumes may need 644/666
            try:
                os.chmod(tmp_name, 0o666) 
            except:
                pass

            os.replace(tmp_name, filepath)
            return True

        except Exception as e:
            # Clean up the temp file left by plan A
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except:
                    pass
            
            # Not the last attempt: wait and retry
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))
                continue
            
            # --- Plan B: fall back to a direct write ---
            # If the atomic replace keeps failing (common on network file systems), write directly
            try:
                print(f"Atomic write failed for {filepath}, using direct write. Error: {e}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception as final_e:
                print(f"Failed to write {filepath} finally: {final_e}")
                return False
    return False

def setup_logger(log_dir, worker_id: int):
    current_date = datetime.now().strftime("%Y%m%d")
    log_filename = f"{log_dir}/worker_{worker_id}_{current_date}.log"
    logger_name = f"worker_{worker_id}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [Worker %(worker_id)s] [PID %(process)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    class WorkerFilter(logging.Filter):
        def __init__(self, worker_id):
            self.worker_id = worker_id
        def filter(self, record):
            record.worker_id = self.worker_id
            return True
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(WorkerFilter(worker_id))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(WorkerFilter(worker_id))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def get_args():
    parser = argparse.ArgumentParser(description="WebFovea - web agent task runner")

    # 1. Input and output
    parser.add_argument("--input", type=str, required=True,
                        help="path to the task JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="output directory (trajectories + logs)")

    # 2. Browser connection
    parser.add_argument("--cdp_url", type=str, nargs='+', required=True,
                        help="browser CDP URLs (one worker per URL)")

    # 3. Model
    parser.add_argument("--model", type=str, default="uitars",
                        help="model name (local vLLM served-model-name or hosted model name)")

    # 4a. Local model: connect to local vLLM servers through vlm_ports
    parser.add_argument("--vlm_ports", type=int, nargs='+', default=[],
                        help="local vLLM server ports (assigned to workers round-robin)")

    # 4b. Hosted model: connect to a remote API through api_base + api_key
    parser.add_argument("--api_base", type=str, default=None,
                        help="hosted model API base URL (with --vlm_ports, its host is combined with each port)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="hosted model API key (prefer the VLM_API_KEY environment variable)")

    # 5. Sampling
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="maximum tokens to generate")

    args = parser.parse_args()

    # === Read config.json (command-line arguments take precedence) ===
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if not args.api_base:
            args.api_base = config.get("api_base")
        if not args.api_key:
            args.api_key = config.get("api_key")
        if args.model == "uitars":  # default value, allow config override
            args.model = config.get("api_model", args.model)
        if not args.vlm_ports:
            ports = config.get("api_ports", [])
            if ports:
                args.vlm_ports = ports
        if args.temperature == 0.7:  # default value
            args.temperature = config.get("temperature", args.temperature)
        if args.top_p == 1.0:  # default value
            args.top_p = config.get("top_p", args.top_p)
        if args.max_tokens == 8192:  # default value
            args.max_tokens = config.get("max_tokens", args.max_tokens)

    # At least one of vlm_ports and api_base is required
    if not args.vlm_ports and not args.api_base:
        parser.error("specify --vlm_ports (local model) or --api_base (hosted model), on the command line or in config.json")

    return args

def print_boot_banner(base_dir, log_dir, cdp_urls, n_tasks, model, api_base):
    """Boot banner: print "which code, which configuration" at the very top of stdout.

    Why: when the agent runs on someone else's machine, **a crash before startup
    (dependencies missing / import failure / broken config) leaves no trace**. No
    worker started, logs/ is empty, and only run.sh's stdout remains. If even part
    of that stdout comes back, these lines answer the three key questions: did it
    start at all, which model and parameters did it run with, and are the numbers
    of tasks and browsers right.

    It does not read git: .git may not exist and git may not be installed, and a
    version number is not worth adding a startup path that can raise. The
    configuration printed below is enough to tell versions apart.

    print only: the loggers are per worker and do not exist yet.
    """
    print("=" * 68, flush=True)
    print("WebFovea BOOT", flush=True)
    try:
        import playwright  # noqa: F401  -- first make sure it imports at all
        # The playwright package has no __version__; ask the package metadata.
        # The version decides whether newer APIs such as select_option()/evaluate_handle() exist.
        from importlib.metadata import version as _pkg_version
        pw_ver = _pkg_version("playwright")
    except Exception as e:
        pw_ver = f"IMPORT FAILED: {e}"     # a missing playwright is the most common startup failure
    print(f"  python           : {platform.python_version()} on {platform.platform()}", flush=True)
    print(f"  playwright       : {pw_ver}", flush=True)
    print(f"  model / api_base : {model} @ {api_base}", flush=True)
    print(f"  config           : {CFG}", flush=True)
    print(f"  tasks            : {n_tasks}", flush=True)
    print(f"  workers (cdp)    : {len(cdp_urls)}", flush=True)
    print(f"  output / logs    : {base_dir} | {log_dir}", flush=True)
    print("BOOT OK", flush=True)          # if this line is missing, startup never completed
    print("=" * 68, flush=True)


if __name__ == "__main__":

    args = get_args()

    # === Input and output paths ===
    BASE_DIR = args.output
    LOG_DIR = os.path.join(args.output, "logs")
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # === Number of workers (= number of CDP URLs) ===
    CDP_URLS = args.cdp_url
    WORKERS = len(CDP_URLS)

    # === VLM ports (local models) ===
    VLM_PORT_LIST = args.vlm_ports  # empty for hosted models

    # === Load tasks ===
    print(f"Loading tasks: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        scribe_json_items = json.load(f)
    print(f"{len(scribe_json_items)} task(s)")

    print_boot_banner(BASE_DIR, LOG_DIR, CDP_URLS, len(scribe_json_items),
                      args.model, args.api_base)

    # === Tasks that already succeeded (resume support) ===
    success_task_map = {}
    if os.path.isdir(BASE_DIR):
        for fname in os.listdir(BASE_DIR):
            rp = os.path.join(BASE_DIR, fname, "result.json")
            if os.path.exists(rp):
                try:
                    with open(rp) as rf:
                        rj = json.load(rf)
                    if rj.get("status") == "SUCCESS":
                        tidx = fname.split("_")[0]
                        success_task_map[tidx] = True
                except Exception:
                    pass
        if success_task_map:
            print(f"Already succeeded: {len(success_task_map)}")

    # === Filter tasks (skip finished ones) ===
    N = len(scribe_json_items)
    indices = []
    skip_cnt = 0
    for i in range(N):
        json_item = scribe_json_items[i]
        task_id = json_item.get("task_id", str(json_item.get("task_index", json_item.get("task_idx"))))
        task_idx = json_item.get("task_index", json_item.get("task_idx"))

        if task_idx in success_task_map:
            skip_cnt += 1
            continue
        website = json_item.get("website", "").strip()
        if not website.startswith("http"):
            website = "https://" + website
        idx_task_id = f"{task_idx}_{task_id}"

        task_dir = f"{BASE_DIR}/{idx_task_id}"
        result_json_path = f"{task_dir}/result.json"
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r', encoding='utf-8') as f:
                result_json_item = json.load(f)
            actions = result_json_item["actions"]
            if len(actions) > 0 and result_json_item["status"] in ["SUCCESS"]:
                skip_cnt += 1
                continue
            else:
                safe_remove_task_dir(task_dir)
        else:
            safe_remove_task_dir(task_dir)
        indices.append(i)

    print(f"Total tasks: {N}, skipped: {skip_cnt}, to run: {len(indices)}")
    if not indices:
        print("All tasks are done!")
        exit(0)
    print(f"Starting from task index {indices[0]}")

    # === Multi-process scheduling: a shared queue instead of static shards ===
    # Why np.array_split was replaced, and what each boundary guards against, is in
    # the module docstring of task_dispatch.py. In short: static sharding turned
    # "one failure stays one failure" from a per-task into a per-worker guarantee.
    manager = mp.Manager()
    completed_tasks = manager.dict()
    # The wall-clock gate counts from process start, which errs on the early side:
    # better to stop early than be killed (a kill leaves neither the current task's
    # result.json nor summary.json).
    # monotonic rather than wall-clock time: it is a system-wide clock, so forked
    # workers read the same timeline and the absolute deadline computed here works
    # in the children as is.
    run_deadline = (time.monotonic() + CFG.run_budget_s) if CFG.run_budget_s else None
    dispatcher = TaskDispatcher(
        indices,
        max_attempts=CFG.max_task_attempts,
        deadline=run_deadline,
        list_factory=manager.list,
        dict_factory=manager.dict,
        lock=manager.Lock(),
    )
    if run_deadline is not None:
        print(f"⏱ Run wall-clock gate: {CFG.run_budget_s}s (after that no new tasks are handed out; running tasks finish)")

    def run_worker(worker_id: int):
        logger = setup_logger(LOG_DIR, worker_id)
        logger.info("Process started, initializing browser and task configuration")

        # === Browser connection ===
        url = CDP_URLS[worker_id]
        print(f"[W{worker_id}] connecting to browser: {url[:80]}...")

        # === Build the agent ===
        if args.vlm_ports:
            # Multi-port local model: take the host from api_base and append vlm_port
            from urllib.parse import urlparse
            vlm_port = VLM_PORT_LIST[worker_id % len(VLM_PORT_LIST)]
            if args.api_base:
                parsed = urlparse(args.api_base)
                scheme = parsed.scheme or "http"
                host = parsed.hostname or "127.0.0.1"
                path = parsed.path.rstrip("/") or "/v1"
                api_url = f"{scheme}://{host}:{vlm_port}{path}"
            else:
                api_url = f"http://127.0.0.1:{vlm_port}/v1"
            api_key = args.api_key or os.getenv("VLM_API_KEY", "EMPTY")
            logger.info(f"Assigned VLM port: {vlm_port}, API: {api_url}")
        else:
            # Hosted API: all workers share api_base
            api_url = args.api_base
            api_key = args.api_key or os.getenv("VLM_API_KEY", "EMPTY")

        agent = UITARSAgent(
            model=args.model,
            api_url=api_url,
            api_key=api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            infer_mode="qwen25vl_normal",  # the action space whose finished(content='xxx') carries the answer
            max_trajectory_length=CFG.max_steps,  # same source as total_steps below, so the two never disagree
        )
        agent.logger = logger

        request_collector = web_controller.RequestCollector()
        # === Initialize the browser ===
        # The reference implementation tried once and returned on failure; a single
        # transient hiccup then cost a whole shard (its tasks never produced a
        # result.json). Now there are two safety nets: retries with backoff, and even
        # if the worker gives up, its tasks stay in the shared pool for other workers.
        p = browser = context = None
        for attempt in range(1, CFG.worker_init_retries + 1):
            try:
                p, browser, context = web_controller.init_playwright_context(url)
                if browser is not None:
                    break
                logger.warning(f"Browser initialization returned nothing (attempt {attempt}/{CFG.worker_init_retries})")
            except Exception as e:
                logger.warning(f"init_playwright_context raised (attempt {attempt}/{CFG.worker_init_retries}): {e}")
                if attempt == CFG.worker_init_retries:
                    traceback.print_exc()
            if attempt < CFG.worker_init_retries:
                time.sleep(min(2 ** attempt, 8))
        if browser is None:
            # This no longer means "these tasks are lost", only "this slot is lost". The
            # parent will try to restart the worker, and other workers take the tasks.
            print(f"[W{worker_id}] browser initialization failed {CFG.worker_init_retries} times; "
                  f"this worker exits (its tasks stay in the shared pool).")
            logger.error("Browser initialization failed for good; worker exits without claiming any task")
            return

        # Consecutive tasks whose page never opened. A risk the shared queue
        # *introduced*: a worker whose browser is dead fails every task it claims within
        # seconds. Under static sharding it only ruined its own share; with a shared
        # queue it can eat everyone's. At the threshold it exits and leaves the pool to
        # healthy workers.
        consecutive_page_failures = 0

        # ── One task, start to finish ────────────────────────────────────
        # claim -> open the start page -> attach the network recorder -> loop
        # [screenshot -> ask the model -> execute the action] until finished / failure /
        # step or time budget exhausted -> write result.json
        # -> finally: write capture.json and mark the task complete with the dispatcher
        while True:
            json_item_i = dispatcher.claim(worker_id)
            if json_item_i is None:
                logger.info("Wall-clock gate closed, no new tasks; worker stopping"
                            if dispatcher.expired() else "Shared pool is empty; worker stopping normally")
                break
            # Whether the network recorder is set up for this task. It decides whether
            # finally may write capture.json: request_collector is reused across tasks and
            # holds the *previous* task's requests until clear(). The page-open failure
            # path never reaches clear(), and writing there would pour the previous
            # task's requests into this task's directory.
            capture_armed = False
            # Reset explicitly on every iteration. These variables used to be checked
            # with `'x' in locals()`, which stays true *across iterations* of the while
            # loop: if the second task raised before assigning base_dir, the fallback
            # would write result.json into the *previous* task's base_dir and overwrite a
            # finished task. Resetting to None and checking `if base_dir` prevents both
            # that and an UnboundLocalError on the first iteration.
            base_dir = None
            task_id = None
            reference_length = None
            try:
                json_item = scribe_json_items[json_item_i]
                task_id = json_item.get("task_id", str(json_item.get("task_index", json_item.get("task_idx"))))
                task = json_item["task"]
                task_idx = json_item.get("task_index", json_item.get("task_idx"))
                website = str(json_item.get("website", "")).lower()
                if task_id in completed_tasks:
                    logger.info(f"Skipping task {json_item_i}/{task_id} (already done)")
                    continue

                attempt_n = dispatcher.attempts_of(json_item_i)
                retry_note = f" (claim #{attempt_n})" if attempt_n > 1 else ""
                logger.info(f"Starting task {json_item_i}/{task_id}{retry_note}: {task[:50]}...")

                base_dir = f"{BASE_DIR}/{task_idx}_{task_id}"

                save_traj_dir = f"{base_dir}/trajectory"
                save_vis_dir = f"{base_dir}/trajectory_visual"
                if not safe_create_directory(base_dir, result_file="result.json"):
                    logger.info(f"Skipping task {json_item_i}/{task_id} (result exists or another process holds it)")
                    continue
                try:
                    page = web_controller.open_page(p, browser, context, url, website)
                    # open_page reconnects by itself when the browser has dropped and
                    # replaces browser/context, but it only returns the page. Without
                    # picking the new ones up, the worker keeps holding dead objects and
                    # every later task reconnects again. Recover them from the page.
                    context = page.context
                    browser = context.browser or browser
                    client = page.context.new_cdp_session(page)
                except Exception as e:
                    # This path produces **no result.json**; the task never really ran.
                    # Return it to the shared pool for another worker (subject to the
                    # max_task_attempts cap); better than producing nothing here.
                    logger.error(f"Task {task_id}: page initialization failed: {e}")
                    requeued = dispatcher.release(worker_id, reason=f"page initialization failed: {type(e).__name__}")
                    consecutive_page_failures += 1
                    if requeued is None:
                        logger.error(f"Task {json_item_i} reached its claim limit and is abandoned (see the abandoned list at the end)")
                    if consecutive_page_failures >= CFG.max_consecutive_page_failures:
                        logger.error(
                            f"Page failed to open for {consecutive_page_failures} tasks in a row; this browser "
                            f"looks dead, so the worker exits and leaves the remaining tasks to others")
                        break
                    continue

                # clear() unbinds the listeners from all of the previous task's pages before
                # emptying the buffer; together with open_page()'s single-tab reset this
                # keeps capture.json to this task's requests only.
                request_collector.clear()
                request_collector.attach(page)
                capture_armed = True        # the buffer now belongs to this task; finally may write it
                logger.info(f"Page opened: {page.url}")
                consecutive_page_failures = 0   # the browser is alive; reset the breaker
                status = ""
                curr_step = 0
                effect_step = 0
                continue_scroll_cnt = 0
                # Echoed from the task if present, otherwise the reference default of 15 (it
                # was once hard-coded to 100, i.e. a made-up baseline). It is only echoed into
                # result.json and **does not set the step budget**: the reference
                # implementation uses ceil(reference_length * 2), but our real constraint is
                # total time, so the budget is CFG.max_steps.
                reference_length = CFG.reference_length_for(json_item)
                total_steps = CFG.max_steps
                task_start = time.time()
                reconnects = 0
                agent.reset()
                web_controller.reset_stall_counters()
                agent.max_trajectory_length = total_steps
                while effect_step < total_steps:
                    if TASK_BUDGET_S > 0 and time.time() - task_start > TASK_BUDGET_S:
                        status = "FAIL_TASK_TIMEOUT"
                        logger.error(
                            f"Task {task_id} exceeded its wall-clock budget of {TASK_BUDGET_S}s "
                            f"({time.time() - task_start:.0f}s used, at step {curr_step}); "
                            f"stopping to protect the time left for other tasks")
                        break
                    image_path = f"{save_traj_dir}/{int(curr_step)}.png"
                    agent.urls.append(page.url)
                    # 50 s x 3 retries meant one hang cost 150 s; at 15 s the worst case is 45 s.
                    screenshot_success = web_controller.save_screenshot(page, savePath=image_path, timeout_ms=15000)
                    if not screenshot_success:
                        # Screenshot timeouts are mostly transient (window occluded, page
                        # briefly stuck). Losing a whole task to one is a bad trade; bring the
                        # page to the front and try once more first.
                        logger.warning(f"Step {curr_step}: screenshot timed out, recovering and retrying once")
                        try:
                            page.bring_to_front()
                        except Exception:
                            pass
                        page.wait_for_timeout(2000)
                        screenshot_success = web_controller.save_screenshot(
                            page, savePath=image_path, timeout_ms=15000)
                    if not screenshot_success:
                        status = "FAIL_SAVE_SCREENSHOT_ERROR"
                        logger.error(f"Step {curr_step}: failed to save screenshot")
                        break
                    screenshot = open(image_path, "rb").read()
                    logger.info(f"### task {json_item_i + 1} of {len(scribe_json_items)}")
                    logger.info(f"### step budget/step/effective step: {total_steps}/{curr_step + 1}/{effect_step + 1}")
                    logger.info(f"### task: {json_item.get('task', '')}")

                    # Feedback on the previous action (rejected / failed / unparseable) goes to the model with the observation
                    prediction, condition = agent.predict(
                        task,
                        {"screenshot": screenshot, "note": web_controller.take_action_note()},
                    )
                    if not agent.get_actions():
                        # The model call failed at step 0 (e.g. three 403s in a row): there is
                        # no action to draw or parse. End as a normal failure and save, rather
                        # than hitting an IndexError and the exception fallback.
                        status = "FAIL"
                        logger.error(f"Step {curr_step}: the model returned no action ({str(prediction)[:120]}); ending the task")
                        break
                    action_str = agent.get_actions()[-1]
                    action = web_controller.parse_action(action_str)
                    draw_point = action.get("coordinate", None)
                    save_debug_image(image_path, bbox=None, point=draw_point, markdown="", action=action_str, save_dir=save_vis_dir)
                    if condition in ["FAIL"]:
                        status = "FAIL"
                        break
                    elif condition in ["DONE"]:
                        status = "SUCCESS"
                        break
                    # ── Timeouts / failures while executing an action ─────────────────
                    # This used to be a bare call: any action that raised bubbled up to the
                    # outer except and **ended the task on the spot**. A click that hit a
                    # navigation race cost a whole task. Now it is downgraded to "this step
                    # did not work": the reason goes back to the model and the loop continues.
                    try:
                        new_page, new_client = web_controller.excute_action(page, client, action)
                    except Exception as act_err:
                        msg = str(act_err)
                        logger.warning(f"Step {curr_step}: action failed - {msg[:200]}")
                        if looks_disconnected(msg) and reconnects < MAX_RECONNECT_PER_TASK:
                            # The browser or tab dropped, e.g. after a long idle period or a slow
                            # model response. Recovery = reopen the task's start page, at most
                            # max_reconnect_per_task times per task, since starting over spends
                            # the steps already used.
                            # With max_reconnect_per_task=0 (the shipped config) this block
                            # **never runs**: reopening the start page amounts to restarting
                            # the task, so a drop ends the task as FAIL_BROWSER_LOST below and
                            # the results are saved as usual. (It never triggered in hundreds of
                            # local runs.)
                            reconnects += 1
                            logger.warning(f"Browser connection lost; recovery #{reconnects}: reopening the start page")
                            try:
                                page = web_controller.open_page(p, browser, context, url, website)
                                context = page.context
                                browser = context.browser or browser
                                client = page.context.new_cdp_session(page)
                                request_collector.attach(page)
                                web_controller.LAST_ACTION_NOTE = (
                                    "The browser connection dropped and the page was reloaded "
                                    "from the task's starting page. Everything you had opened is "
                                    "gone - start again from what is on screen now.")
                                curr_step += 1
                                continue
                            except Exception as rec_err:
                                logger.error(f"Recovery after disconnect failed: {rec_err}")
                                status = "FAIL_BROWSER_LOST"
                                break
                        if looks_disconnected(msg):
                            # Out of recoveries and still disconnected = the browser is really
                            # gone. Going on would burn the remaining steps and tokens on a dead
                            # session; cut the loss now and leave the time to later tasks.
                            logger.error("No recoveries left; the browser is unusable, ending the task")
                            status = "FAIL_BROWSER_LOST"
                            break
                        # Any other failure: this step is void; tell the model why and carry on
                        web_controller.LAST_ACTION_NOTE = (
                            f"The previous action failed to execute and had no effect "
                            f"({msg[:180]}). Try a different element or a different approach.")
                        new_page, new_client = page, client

                    if new_page != page:
                        logger.info(f"Page switched: {page.url} -> {new_page.url}")
                        page = new_page
                        client = page.context.new_cdp_session(page)
                        request_collector.attach(page)

                    if "scroll" not in action_str:
                        continue_scroll_cnt = 0
                        # click_many is charged one step per square clicked; see step_cost
                        effect_step += web_controller.step_cost(action)
                    else:
                        continue_scroll_cnt += 1
                        # This used to count consecutive scrolls only, without checking
                        # whether the page changed. One task was cut off while **reading a
                        # long table screen by screen**, with new content every time: the
                        # breaker fired one step before the agent could sum up the table, and
                        # it ended with an empty answer, throwing away everything it had read.
                        #
                        # The test is now "consecutive scrolls AND no change": when the agent
                        # is truly idle, web_controller's fingerprint comparison increments
                        # _NO_CHANGE_STREAK, while reading a table changes the fingerprint
                        # every time and resets the streak. Only both together count as idle.
                        if continue_scroll_cnt >= 10:
                            if web_controller.no_change_streak() >= 3:
                                logger.info("10 scrolls in a row with no page change: idle, leaving the loop")
                                status = "FAIL_SCROLLDOWN"
                                break
                            # The page is still changing = reading long content; do not stop it, but nudge
                            if continue_scroll_cnt == 10 or continue_scroll_cnt % 5 == 0:
                                logger.info(f"{continue_scroll_cnt} scrolls in a row, page still changing; "
                                            f"not idle, sending a reminder only")
                                web_controller.LAST_ACTION_NOTE = (
                                    f"You have scrolled {continue_scroll_cnt} times in a row. "
                                    f"The page is still changing, so this is not wasted - but if "
                                    f"you already have what you need, stop scrolling and answer. "
                                    f"If you are hunting for something you can name, use "
                                    f"find_text(text='...') instead of scrolling further.")
                    curr_step += 1

                if status == "FAIL_SAVE_SCREENSHOT_ERROR":
                    # The reference implementation simply continued here, so the task produced
                    # no result.json at all: trajectory, thoughts and any agent_answer already
                    # obtained were lost, and the failure could not be analysed afterwards.
                    # Now the result is written as usual, with the reason in status.
                    logger.info("Task ended early on a screenshot failure; still writing result.json to keep the trajectory")

                responses = agent.get_history_responses()
                thoughts = agent.get_thoughts()
                actions = agent.get_actions()
                final_result_response = thoughts[-1] if thoughts else ""
                urls = agent.get_urls()
                result_item = {
                    "task_idx": json_item.get("task_idx", ""),
                    "task_id": task_id,
                    "task": task,
                    "website": website,
                    "status": status,
                    "reference_length": reference_length,
                    "predict_length": len(actions),
                    "agent_answer": extract_agent_answer(actions),
                    "final_result_response": final_result_response,
                    "actions": actions,
                    "thoughts": thoughts,
                    "history_resps": responses,
                    "urls": urls
                }
                if not safe_save_results(base_dir, result_item):
                    logger.error(f"Failed to save the result file: {base_dir}")

                completed_tasks[task_id] = worker_id
                agent.reset()
                logger.info(f"### task {json_item_i + 1} of {len(scribe_json_items)} done")
            except Exception as e:
                logger.error(f"Exception while processing task {json_item_i}: {str(e)}", exc_info=True)
                # Fallback save: this path used to simply continue, losing the trajectory,
                # thoughts and any agent_answer already extracted. A page closed under the
                # agent ("Target page has been closed") once wiped out two tasks this way.
                # Every exit path must leave a result.json.
                try:
                    if base_dir and task_id:
                        _actions = agent.get_actions()
                        _thoughts = agent.get_thoughts()
                        safe_save_results(base_dir, {
                            "task_idx": json_item.get("task_idx", ""),
                            "task_id": task_id,
                            "task": task,
                            "website": website,
                            "status": f"FAIL_EXCEPTION: {type(e).__name__}",
                            "reference_length": (reference_length if reference_length
                                                 else CFG.reference_length_for(json_item)),
                            "predict_length": len(_actions),
                            "agent_answer": extract_agent_answer(_actions),
                            "final_result_response": _thoughts[-1] if _thoughts else "",
                            "actions": _actions,
                            "thoughts": _thoughts,
                            "history_resps": agent.get_history_responses(),
                            "urls": agent.get_urls(),
                        })
                        logger.info("Wrote result.json from the exception fallback")
                except Exception as e2:
                    logger.error(f"The exception fallback failed to save too: {e2}")
                try:
                    agent.reset()
                except Exception:
                    pass
                continue
            finally:
                # capture.json is written here and only here, which covers the normal,
                # exception and breaker exits alike. The reference implementation wrote it
                # on the normal path only, and tasks that took the exception path were
                # missing it. For the armed gate and "never raise", see save_capture_json.
                save_capture_json(request_collector, base_dir, capture_armed, logger)

                # Must be in finally. The task has an outcome (every exit path above wrote
                # result.json) and is **never re-sent**; re-running it would only wipe its
                # trajectory and start over. Without this call the worker keeps the task as
                # inflight, and when it later exits normally the parent mistakes that for a
                # mid-task crash and re-sends the task. On the page-open failure path the
                # task was already released, so this call is a no-op there.
                dispatcher.complete(worker_id)

        try:
            context.close()
            p.stop()
            logger.info(f"Worker {worker_id} cleaned up")
        except Exception as e:
            logger.error(f"Worker {worker_id} cleanup failed: {str(e)}")

    # === Start workers ===
    # With fewer tasks than workers there is no need to connect every browser (the
    # reference implementation got the same effect by skipping empty shards). Small
    # test runs hit this most often.
    n_workers = min(WORKERS, len(indices))
    print(f"Starting {n_workers} process(es) on a shared pool of {dispatcher.total} task(s)")

    def start_worker(wid: int):
        proc = mp.Process(target=run_worker, args=(wid,))
        proc.start()
        return proc

    print("\nMonitoring worker processes...")
    respawns = supervise_workers(
        dispatcher, range(n_workers), start_worker,
        max_respawns=CFG.max_worker_respawns,
        status_extra=lambda: f"done: {len(completed_tasks)}",
    )

    # All workers are gone but tasks remain: a pure loss (not even a result.json).
    # Print them explicitly: a few quietly missing tasks are the hardest failure to
    # notice afterwards, especially when only a total score comes back.
    left = dispatcher.drain_pending(
        f"wall-clock gate of {CFG.run_budget_s}s closed before it started" if dispatcher.expired()
        else "all workers exited; nobody claimed it")
    if left:
        print(f"🔴 {len(left)} task(s) never claimed: {left}")

    print(f"\nDone. Total: {len(completed_tasks)}/{len(indices)}")
    summary_file = os.path.join(LOG_DIR, "summary.json")
    # The abandoned list must be saved. When only a total score comes back, it is the
    # only clue to why a task has no result at all.
    abandoned = dispatcher.abandoned()
    summary = {
        "total_tasks": len(indices),
        "completed_tasks": len(completed_tasks),
        "completion_rate": f"{len(completed_tasks)/len(indices)*100:.2f}%",
        "completed_task_ids": list(completed_tasks.keys()),
        "abandoned_count": len(abandoned),
        "abandoned": [{"task_index": i, "reason": r} for i, r in abandoned],
        "worker_respawns": dict(respawns),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if abandoned:
        print(f"🔴 {len(abandoned)} task(s) abandoned (no result.json):")
        for i, r in abandoned:
            print(f"    task_index={i}  reason: {r}")
    safe_write_json(summary_file, summary)
    print(f"Summary: {summary_file}")
