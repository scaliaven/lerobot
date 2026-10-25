"""Run LIBERO locally, with every model action produced by a remote OpenVLA server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from remote_protocol import SUITES, RemotePolicy

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NOOP = [0, 0, 0, 0, 0, 0, -1]
LOGGER = logging.getLogger(__name__)

PREVIEW_HTML = """<!doctype html><meta charset="utf-8"><title>LIBERO remote inference</title>
<style>body{background:#18212b;color:#eee;font:16px system-ui;max-width:820px;margin:30px auto}
img{width:512px;max-width:100%;image-rendering:auto}pre{white-space:pre-wrap}</style>
<h1>LIBERO · remote OpenVLA</h1><img id="frame" alt="Waiting for simulation frame">
<pre id="status">Waiting for status…</pre>
<script>async function refresh(){try{
const s=await fetch('latest.json?t='+Date.now()).then(r=>r.json());
document.getElementById('status').textContent=JSON.stringify(s,null,2);
document.getElementById('frame').src='latest.png?t='+Date.now();
}catch(e){document.getElementById('status').textContent='Waiting for simulation…';}}
setInterval(refresh,750);refresh();</script>"""


def configure_libero(config_dir: Path) -> None:
    """Avoid LIBERO's interactive first-import prompt without modifying the user's existing config."""
    if "LIBERO_CONFIG_PATH" not in os.environ:
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    destination = Path(os.environ["LIBERO_CONFIG_PATH"])
    if (destination / "config.yaml").exists():
        return
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO is missing. Run the Linux client setup or use the Docker client.")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    paths = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(destination / "datasets"),
        "assets": str(root / "assets"),
    }
    destination.mkdir(parents=True, exist_ok=True)
    # JSON is a subset of YAML and LIBERO reads this with yaml.FullLoader.
    (destination / "config.yaml").write_text(json.dumps(paths, indent=2))


def upright(image: np.ndarray) -> np.ndarray:
    """LIBERO renders upside down; rotate a copy for human viewing."""
    return image[::-1, ::-1].copy()


class Artifacts:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text(PREVIEW_HTML)

    def publish(self, image: np.ndarray | None, status: dict) -> None:
        """Write the status, and the upright frame unless `image` is None (frame unchanged)."""
        if image is not None:
            Image.fromarray(upright(image)).save(self.directory / "frame.tmp", format="PNG")
            (self.directory / "frame.tmp").replace(self.directory / "latest.png")
        (self.directory / "status.tmp").write_text(json.dumps(status, indent=2, allow_nan=False))
        (self.directory / "status.tmp").replace(self.directory / "latest.json")


@contextmanager
def preview_server(directory: Path, host: str, port: int):
    if port == 0:
        yield
        return
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory.resolve()))
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    LOGGER.info("Preview: http://127.0.0.1:%s (serving only %s)", port, directory)
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_episode(
    env: Any,
    initial_state: Any,
    policy: Any,
    task: str,
    episode: int,
    max_steps: int,
    settle_steps: int,
    artifacts: Artifacts,
    metadata: dict,
    writer: Any = None,
) -> dict:
    """Simulation time advances only after an action reply; never load a model locally."""
    env.seed(0)
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(settle_steps):
        observation, _, _, _ = env.step(NOOP)
    success, steps = False, 0
    run_id = uuid.uuid4().hex
    trace_path = artifacts.directory / f"episode-{episode:03d}.jsonl"
    with trace_path.open("w") as trace:
        for step in range(max_steps):
            image = observation["agentview_image"]
            status = {"episode": episode, "step": step, "task": task, **metadata}
            # After the first step this frame was already published as the previous step's result.
            artifacts.publish(image if step == 0 else None, {**status, "phase": "waiting for remote action"})
            if writer is not None:
                writer.append_data(upright(image))
            request_id = f"{run_id}:{step}"
            start = time.perf_counter()
            action, reply = policy.predict(image, task, request_id)
            round_trip_ms = (time.perf_counter() - start) * 1000
            # RemotePolicy.predict already validated the action. Server already converts the gripper; do not invert it a second time here.
            observation, reward, done, _ = env.step(action)
            steps = step + 1
            success = bool(env.check_success())
            record = {
                **status,
                "phase": "action applied",
                "request_id": request_id,
                "action": action,
                "reward": float(reward),
                "success": success,
                "round_trip_ms": round_trip_ms,
                "inference_ms": reply.get("inference_ms"),
            }
            trace.write(json.dumps(record, allow_nan=False) + "\n")
            trace.flush()
            artifacts.publish(observation["agentview_image"], record)
            LOGGER.info(
                "Episode %s step %s: %.0f ms round-trip; success=%s", episode, step, round_trip_ms, success
            )
            if done or success:
                break
        if writer is not None:
            writer.append_data(upright(observation["agentview_image"]))
    return {"episode": episode, "steps": steps, "success": success, "task": task}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--suite", choices=SUITES, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/remote-libero") / time.strftime("%Y%m%d-%H%M%S")
    )
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--preview-port", type=int, default=8090, help="0 disables live preview")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.settle_steps < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("episodes/max-steps must be positive and settle-steps must be nonnegative")
    if sys.platform != "linux":
        parser.error(
            "LIBERO is supported on Linux. On macOS, use run.sh docker-client (CPU-rendered Linux container)."
        )
    logging.basicConfig(level=logging.INFO)
    policy = RemotePolicy(args.endpoint, args.suite, args.timeout)
    metadata = {} if args.list_tasks else policy.check()
    LOGGER.info("Remote server: %s", metadata)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    configure_libero(args.output_dir / ".libero")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.list_tasks:
        for index in range(suite.n_tasks):
            print(f"{index}: {suite.get_task(index).language}")
        return
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error(f"task-id must be in [0, {suite.n_tasks - 1}]")
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)
    if args.episodes > len(states):
        parser.error(f"Only {len(states)} fixed initial states are available for this task")
    artifacts = Artifacts(args.output_dir)
    summaries = []
    with preview_server(args.output_dir, args.preview_host, args.preview_port):
        for episode in range(args.episodes):
            # A fresh environment per trial keeps fixture placement from leaking between resets.
            env = OffScreenRenderEnv(
                bddl_file_name=str(
                    Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                ),
                camera_heights=256,
                camera_widths=256,
                control_freq=20,
            )
            writer = None
            try:
                if not args.no_video:
                    import imageio.v2 as imageio

                    writer = imageio.get_writer(str(args.output_dir / f"episode-{episode:03d}.mp4"), fps=20)
                summary = run_episode(
                    env,
                    states[episode],
                    policy,
                    task.language,
                    episode,
                    args.max_steps or MAX_STEPS[args.suite],
                    args.settle_steps,
                    artifacts,
                    metadata,
                    writer,
                )
                summaries.append(summary)
                (args.output_dir / "summary.json").write_text(
                    json.dumps({"server": metadata, "episodes": summaries}, indent=2)
                )
            finally:
                if writer is not None:
                    writer.close()
                env.close()
    LOGGER.info("Finished. Artifacts: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
