#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discrete terminal keyboard control for a calibrated SO-101 follower (macOS/Linux)."""

import argparse
import math
import sys
import time
from contextlib import suppress
from queue import Empty, Full, Queue
from threading import Event

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.keyboard_input import TerminalKeyListener

KEYS = {
    "a": ("shoulder_pan", 1),
    "d": ("shoulder_pan", -1),
    "w": ("shoulder_lift", 1),
    "s": ("shoulder_lift", -1),
    "r": ("elbow_flex", 1),
    "f": ("elbow_flex", -1),
    "t": ("wrist_flex", 1),
    "g": ("wrist_flex", -1),
    "y": ("wrist_roll", 1),
    "h": ("wrist_roll", -1),
    "o": ("gripper", 1),
    "p": ("gripper", -1),
}
HELP = "A/D: base | W/S: shoulder | R/F: elbow | T/G: wrist bend | Y/H: wrist roll | O/P: gripper +/-"
MAX_KEY_AGE_S = 0.25
# How far a commanded target may run ahead of the measured position (e.g. a joint sagging under load).
MAX_LEAD = 5.0


def joint_limits(robot: SO101Follower) -> dict[str, tuple[float, float]]:
    """Convert the driver's calibrated encoder endpoints to its degree coordinates."""
    limits = {}
    for name, motor in robot.bus.motors.items():
        calibration = robot.calibration[name]
        if calibration.range_max <= calibration.range_min:
            raise ValueError(f"Invalid calibration range for {name}")
        if name == "gripper":
            limits[name] = (0.0, 100.0)
        else:
            resolution = robot.bus.model_resolution_table[motor.model] - 1
            half_range = (calibration.range_max - calibration.range_min) * 180.0 / resolution
            limits[name] = (-half_range, half_range)
    return limits


def action_for_key(key, observation, limits, step=1.0, gripper_step=2.0, targets=None):
    """Nudge one joint from its last sent target, kept within MAX_LEAD of the measured position."""
    binding = KEYS.get(key.lower())
    if binding is None:
        return {}
    joint, direction = binding
    name = f"{joint}.pos"
    position = float(observation[name])
    if not math.isfinite(position):
        raise ValueError(f"Non-finite position for {joint}")
    low, high = limits[joint]
    # Degree readings are not clamped by the driver, so a joint resting at its stop can read a bit outside.
    position = min(high, max(low, position))
    base = (targets or {}).get(name, position)
    base = min(position + MAX_LEAD, max(position - MAX_LEAD, base))
    amount = gripper_step if joint == "gripper" else step
    target = min(high, max(low, base + direction * amount))
    return {name: target}


def teleoperate(robot: SO101Follower, step=1.0, gripper_step=2.0):
    stopped = Event()
    pending = Queue(maxsize=1)

    def on_key(key):
        key = key.lower()
        if key in ("q", "esc"):
            stopped.set()
        elif key in KEYS:
            # Keep at most one command, so key repeat cannot build a movement backlog.
            with suppress(Empty):
                pending.get_nowait()
            with suppress(Full):
                pending.put_nowait((key, time.monotonic()))

    listener = TerminalKeyListener(on_key)
    targets = {}
    try:
        robot.connect()
        limits = joint_limits(robot)
        print(HELP)
        print("One small step per key event (including terminal key repeat).")
        print("Q / Esc / Ctrl-C exits and releases torque. Support the arm before exiting.", flush=True)
        listener.start()
        while not stopped.is_set():
            try:
                key, pressed_at = pending.get(timeout=0.05)
            except Empty:
                continue
            if stopped.is_set() or time.monotonic() - pressed_at > MAX_KEY_AGE_S:
                continue
            observation = robot.get_observation()
            action = action_for_key(key, observation, limits, step, gripper_step, targets)
            if not stopped.is_set() and time.monotonic() - pressed_at <= MAX_KEY_AGE_S:
                robot.send_action(action)
                targets.update(action)
    finally:
        try:
            listener.stop()
        finally:
            if robot.is_connected:
                robot.disconnect()


def small_step(text):
    value = float(text)
    if not math.isfinite(value) or not 0 < value <= 5:
        raise argparse.ArgumentTypeError("Choose a finite step greater than 0 and at most 5")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=HELP)
    parser.add_argument("--port", required=True, help="Follower serial port")
    parser.add_argument("--id", required=True, help="Reuse the follower's calibration ID")
    parser.add_argument(
        "--step", type=small_step, default=1.0, help="Degrees per joint key event (default: 1)"
    )
    parser.add_argument(
        "--gripper-step",
        type=small_step,
        default=2.0,
        help="Gripper percentage points per event (default: 2)",
    )
    args = parser.parse_args()
    if sys.platform == "win32" or not sys.stdin.isatty():
        parser.error("Run in an interactive macOS/Linux terminal; keep that terminal focused")
    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            use_degrees=True,
            # Measured from the present position, so it must also allow the commanded-target lead.
            max_relative_target=max(args.step, args.gripper_step) + MAX_LEAD,
            disable_torque_on_disconnect=True,
        )
    )
    if not robot.calibration:
        parser.error("No saved calibration for this ID. Calibrate the follower first or use its existing ID")
    with suppress(KeyboardInterrupt):
        teleoperate(robot, args.step, args.gripper_step)


if __name__ == "__main__":
    main()
