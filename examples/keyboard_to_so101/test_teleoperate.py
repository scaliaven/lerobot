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

"""Hardware-free checks: run `python -m unittest examples.keyboard_to_so101.test_teleoperate` from the repo root."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from examples.keyboard_to_so101 import teleoperate as example


def fake_robot():
    robot = Mock()
    robot.is_connected = False
    robot.calibration = {
        joint: SimpleNamespace(range_min=1000, range_max=3000) for joint, _ in example.KEYS.values()
    }
    robot.bus.motors = {joint: SimpleNamespace(model="sts3215") for joint in robot.calibration}
    robot.bus.model_resolution_table = {"sts3215": 4096}
    robot.get_observation.return_value = {f"{joint}.pos": 0.0 for joint in robot.calibration}

    def connect():
        robot.is_connected = True

    def disconnect():
        robot.is_connected = False

    robot.connect.side_effect = connect
    robot.disconnect.side_effect = disconnect
    return robot


class KeyboardControlTests(unittest.TestCase):
    def setUp(self):
        self.robot = fake_robot()
        self.limits = example.joint_limits(self.robot)

    def test_targets_use_observed_position_and_calibrated_endpoint(self):
        high = self.limits["shoulder_lift"][1]
        observation = {"shoulder_lift.pos": high - 0.1}
        self.assertEqual(example.action_for_key("w", observation, self.limits), {"shoulder_lift.pos": high})
        self.assertEqual(
            example.action_for_key("s", observation, self.limits), {"shoulder_lift.pos": high - 1.1}
        )

    def test_opposite_keys_and_uppercase(self):
        observation = self.robot.get_observation()
        self.assertEqual(example.action_for_key("A", observation, self.limits), {"shoulder_pan.pos": 1.0})
        self.assertEqual(example.action_for_key("d", observation, self.limits), {"shoulder_pan.pos": -1.0})

    def test_gripper_uses_percentage_units_and_its_own_step(self):
        self.assertEqual(
            example.action_for_key("o", {"gripper.pos": 99.0}, self.limits), {"gripper.pos": 100.0}
        )
        self.assertEqual(example.action_for_key("p", {"gripper.pos": 1.0}, self.limits), {"gripper.pos": 0.0})

    def test_unknown_keys_do_not_command_motion(self):
        self.assertEqual(example.action_for_key("up", {}, {}), {})

    def test_invalid_sensor_positions_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                example.action_for_key("w", {"shoulder_lift.pos": value}, self.limits)

    def test_reading_past_calibrated_endpoint_is_clamped(self):
        high = self.limits["shoulder_lift"][1]
        observation = {"shoulder_lift.pos": high + 1}
        self.assertEqual(example.action_for_key("w", observation, self.limits), {"shoulder_lift.pos": high})
        self.assertEqual(
            example.action_for_key("s", observation, self.limits), {"shoulder_lift.pos": high - 1}
        )

    def test_step_builds_on_last_target_when_joint_sags(self):
        targets = {"shoulder_lift.pos": 10.0}
        observation = {"shoulder_lift.pos": 8.5}
        self.assertEqual(
            example.action_for_key("w", observation, self.limits, targets=targets),
            {"shoulder_lift.pos": 11.0},
        )
        self.assertEqual(
            example.action_for_key("s", observation, self.limits, targets=targets), {"shoulder_lift.pos": 9.0}
        )

    def test_target_lead_over_measured_position_is_bounded(self):
        targets = {"shoulder_lift.pos": 30.0}
        observation = {"shoulder_lift.pos": 0.0}
        self.assertEqual(
            example.action_for_key("w", observation, self.limits, targets=targets),
            {"shoulder_lift.pos": example.MAX_LEAD + 1},
        )

    def test_invalid_calibration_rejected(self):
        self.robot.calibration["gripper"].range_max = 1000
        with self.assertRaises(ValueError):
            example.joint_limits(self.robot)

    def test_invalid_cli_steps_rejected(self):
        for text in ("nan", "inf", "0", "-1", "6"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                example.small_step(text)

    def test_movement_then_quit_disconnects(self):
        with patch.object(example, "TerminalKeyListener") as listener_type:
            listener = listener_type.return_value
            listener.start.side_effect = lambda: listener_type.call_args.args[0]("w")
            self.robot.send_action.side_effect = lambda _: listener_type.call_args.args[0]("q")
            example.teleoperate(self.robot)
            self.robot.send_action.assert_called_once_with({"shoulder_lift.pos": 1.0})
            listener.stop.assert_called_once()
            self.robot.disconnect.assert_called_once()

    def test_repeated_keys_accumulate_from_sent_targets(self):
        with patch.object(example, "TerminalKeyListener") as listener_type:
            listener_type.return_value.start.side_effect = lambda: listener_type.call_args.args[0]("w")
            sent = []

            def send(action):
                sent.append(action)
                listener_type.call_args.args[0]("w" if len(sent) < 2 else "q")

            self.robot.send_action.side_effect = send
            example.teleoperate(self.robot)
            # The fake arm never moves, so the second step must build on the first target.
            self.assertEqual(sent, [{"shoulder_lift.pos": 1.0}, {"shoulder_lift.pos": 2.0}])

    def test_quit_takes_priority_over_pending_movement(self):
        with patch.object(example, "TerminalKeyListener") as listener_type:

            def start():
                on_key = listener_type.call_args.args[0]
                on_key("w")
                on_key("q")

            listener_type.return_value.start.side_effect = start
            example.teleoperate(self.robot)
            self.robot.send_action.assert_not_called()
            self.robot.disconnect.assert_called_once()

    def test_quit_during_observation_discards_key(self):
        with (
            patch.object(example, "TerminalKeyListener") as listener_type,
            patch.object(example.time, "monotonic") as clock,
        ):
            listener_type.return_value.start.side_effect = lambda: listener_type.call_args.args[0]("w")
            clock.return_value = 0.0

            def observe():
                clock.return_value = example.MAX_KEY_AGE_S + 1
                listener_type.call_args.args[0]("q")
                return {"shoulder_lift.pos": 0.0}

            self.robot.get_observation.side_effect = observe
            example.teleoperate(self.robot)
            self.robot.send_action.assert_not_called()

    def test_slow_observation_discards_stale_key(self):
        with (
            patch.object(example, "TerminalKeyListener"),
            patch.object(example, "Queue") as queue_type,
            patch.object(example.time, "monotonic", side_effect=[0.0, example.MAX_KEY_AGE_S + 1]),
        ):
            queue_type.return_value.get.side_effect = [("w", 0.0), KeyboardInterrupt]
            with self.assertRaises(KeyboardInterrupt):
                example.teleoperate(self.robot)
            self.robot.get_observation.assert_called_once()
            self.robot.send_action.assert_not_called()
            self.robot.disconnect.assert_called_once()

    def test_key_repeat_keeps_only_latest_pending_command(self):
        with patch.object(example, "TerminalKeyListener") as listener_type:

            def start():
                on_key = listener_type.call_args.args[0]
                for key in ("w", "w", "s"):
                    on_key(key)

            listener_type.return_value.start.side_effect = start
            self.robot.send_action.side_effect = lambda _: listener_type.call_args.args[0]("q")
            example.teleoperate(self.robot)
            self.robot.send_action.assert_called_once_with({"shoulder_lift.pos": -1.0})

    def test_sensor_failure_restores_listener_and_disconnects(self):
        with patch.object(example, "TerminalKeyListener") as listener_type:
            listener_type.return_value.start.side_effect = lambda: listener_type.call_args.args[0]("w")
            self.robot.get_observation.side_effect = RuntimeError("serial read failed")
            with self.assertRaisesRegex(RuntimeError, "serial read failed"):
                example.teleoperate(self.robot)
            self.robot.send_action.assert_not_called()
            listener_type.return_value.stop.assert_called_once()
            self.robot.disconnect.assert_called_once()

    def test_keyboard_interrupt_disconnects(self):
        with patch.object(example, "TerminalKeyListener") as listener_type:
            listener_type.return_value.start.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                example.teleoperate(self.robot)
            listener_type.return_value.stop.assert_called_once()
            self.robot.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
