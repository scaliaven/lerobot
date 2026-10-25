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

from unittest.mock import Mock

import pytest

pygame = pytest.importorskip("pygame")

from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig  # noqa: E402
from lerobot.teleoperators.gamepad.gamepad_utils import SDLGamepadController  # noqa: E402
from lerobot.teleoperators.utils import TeleopEvents  # noqa: E402


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setattr(pygame.event, "pump", lambda: None)
    result = SDLGamepadController()
    result.controller = Mock()
    result.controller.attached.return_value = True
    result.controller.get_button.return_value = False
    result.controller.get_axis.return_value = 0
    return result


def test_neutral_input_does_not_enable_movement(reader):
    reader.update()
    assert not reader.should_intervene()
    assert reader.get_deltas() == (0.0, 0.0, 0.0)
    assert reader.gripper_command() == "stay"


def test_right_shoulder_is_a_polled_deadman(reader):
    reader.controller.get_button.side_effect = lambda key: key == pygame.CONTROLLER_BUTTON_RIGHTSHOULDER
    reader.update()
    assert reader.should_intervene()
    reader.controller.get_button.side_effect = lambda key: False
    reader.update()
    assert not reader.should_intervene()


@pytest.mark.parametrize(
    "axis, expected",
    [
        (pygame.CONTROLLER_AXIS_LEFTY, (0.5, 0.0, 0.0)),
        (pygame.CONTROLLER_AXIS_LEFTX, (0.0, 0.5, 0.0)),
        (pygame.CONTROLLER_AXIS_RIGHTY, (0.0, 0.0, 0.5)),
    ],
)
def test_standardized_stick_axes(reader, axis, expected):
    reader.controller.get_axis.side_effect = lambda key: -16384 if key == axis else 0
    assert reader.get_deltas() == expected


@pytest.mark.parametrize(
    "dpad, expected",
    [
        (pygame.CONTROLLER_BUTTON_DPAD_UP, (-1, 0)),
        (pygame.CONTROLLER_BUTTON_DPAD_DOWN, (1, 0)),
        (pygame.CONTROLLER_BUTTON_DPAD_LEFT, (0, -1)),
        (pygame.CONTROLLER_BUTTON_DPAD_RIGHT, (0, 1)),
    ],
)
def test_dpad_jogs_wrist_only_while_r_is_held(reader, dpad, expected):
    reader.controller.get_button.side_effect = lambda key: key == dpad
    reader.update()
    assert reader.get_wrist_deltas() == (0, 0)
    reader.controller.get_button.side_effect = lambda key: key in (
        dpad,
        pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
    )
    reader.update()
    assert reader.get_wrist_deltas() == expected


def test_deadzone_suppresses_drift(reader):
    reader.controller.get_axis.return_value = 1000
    assert reader.get_deltas() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "axis, command",
    [(pygame.CONTROLLER_AXIS_TRIGGERLEFT, "open"), (pygame.CONTROLLER_AXIS_TRIGGERRIGHT, "close")],
)
def test_triggers_supply_existing_gripper_commands(reader, axis, command):
    reader.controller.get_axis.side_effect = lambda key: 32767 if key == axis else 0
    reader.update()
    assert reader.gripper_command() == command


def test_simultaneous_triggers_cancel(reader):
    reader.controller.get_axis.return_value = 32767
    reader.update()
    assert reader.gripper_command() == "stay"


def test_minus_exits(reader):
    reader.controller.get_button.side_effect = lambda key: key == pygame.CONTROLLER_BUTTON_BACK
    with pytest.raises(KeyboardInterrupt):
        reader.update()


def test_disconnect_clears_deadman_and_raises(reader):
    reader.intervention_flag = True
    reader.controller.attached.return_value = False
    with pytest.raises(ConnectionError):
        reader.update()
    assert not reader.should_intervene()


def test_held_episode_button_generates_only_one_event(reader):
    reader.controller.get_button.side_effect = lambda key: key == pygame.CONTROLLER_BUTTON_Y
    reader.update()
    assert reader.get_episode_end_status() == TeleopEvents.SUCCESS
    reader.update()
    assert reader.get_episode_end_status() is None


def test_conflicting_backends_rejected():
    with pytest.raises(ValueError):
        GamepadTeleopConfig(hidapi_fallback=True, use_sdl_controller=True)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_existing_entrypoint_disconnects_on_failure(monkeypatch, failure):
    pytest.importorskip("datasets")
    from lerobot.rl import gym_manipulator

    env, teleop = Mock(), Mock()
    monkeypatch.setattr(gym_manipulator, "make_robot_env", lambda cfg: (env, teleop))
    monkeypatch.setattr(gym_manipulator, "make_processors", lambda *args, **kwargs: (Mock(), Mock()))
    monkeypatch.setattr(gym_manipulator, "control_loop", Mock(side_effect=failure))
    config = Mock(mode=None)
    if failure is KeyboardInterrupt:
        gym_manipulator.main.__wrapped__(config)
    else:
        with pytest.raises(RuntimeError):
            gym_manipulator.main.__wrapped__(config)
    env.close.assert_called_once()
    teleop.disconnect.assert_called_once()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_startup_failure_cleans_up_before_environment_exists(monkeypatch, cleanup_fails):
    pytest.importorskip("datasets")
    from lerobot.rl import gym_manipulator

    robot, teleop = Mock(cameras={}), Mock()
    robot.is_connected = True
    teleop.is_connected = True
    if cleanup_fails:
        robot.disconnect.side_effect = ConnectionError("cleanup also failed")
    monkeypatch.setattr(gym_manipulator, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(gym_manipulator, "make_teleoperator_from_config", lambda cfg: teleop)
    monkeypatch.setattr(
        gym_manipulator, "RobotEnv", Mock(side_effect=ConnectionError("Lock on id_=5: no status packet"))
    )
    with pytest.raises(ConnectionError, match="Lock on id_=5"):
        gym_manipulator.make_robot_env(Mock(name="real_robot"))
    robot.disconnect.assert_called_once()
    teleop.disconnect.assert_called_once()


def test_startup_failure_releases_bus_when_a_camera_failed(monkeypatch):
    pytest.importorskip("datasets")
    from lerobot.rl import gym_manipulator

    camera = Mock(is_connected=False)
    robot, teleop = Mock(cameras={"front": camera}), Mock(is_connected=False)
    robot.is_connected = False  # The bus opened but a camera did not.
    robot.bus.is_connected = True
    monkeypatch.setattr(gym_manipulator, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(gym_manipulator, "make_teleoperator_from_config", lambda cfg: teleop)
    monkeypatch.setattr(gym_manipulator, "RobotEnv", Mock(side_effect=ConnectionError("camera failed")))
    with pytest.raises(ConnectionError):
        gym_manipulator.make_robot_env(Mock(name="real_robot"))
    robot.bus.disconnect.assert_called_once_with(robot.config.disable_torque_on_disconnect)
    robot.disconnect.assert_not_called()
    camera.disconnect.assert_not_called()
    teleop.disconnect.assert_not_called()


def test_cleanup_closes_port_and_cameras_when_torque_disable_fails():
    pytest.importorskip("datasets")
    from lerobot.rl import gym_manipulator

    camera = Mock(is_connected=True)
    robot = Mock(cameras={"front": camera}, is_connected=False)
    robot.bus.is_connected = True
    robot.bus.disconnect.side_effect = [ConnectionError("torque write timed out"), None]
    gym_manipulator.release_robot(robot)
    assert robot.bus.disconnect.call_args_list[-1].kwargs == {"disable_torque": False}
    camera.disconnect.assert_called_once()


@pytest.mark.parametrize("record, push", [(True, True), (True, False), (False, False)])
def test_stop_during_recording_still_finalizes_dataset(monkeypatch, record, push):
    pytest.importorskip("datasets")
    from lerobot.processor import TransitionKey
    from lerobot.rl import gym_manipulator

    dataset = Mock()
    monkeypatch.setattr(gym_manipulator.LeRobotDataset, "create", Mock(return_value=dataset))
    monkeypatch.setattr(
        gym_manipulator, "reset_and_build_transition", lambda *args: {TransitionKey.OBSERVATION: {}}
    )
    # The SDL controller's Minus button raises KeyboardInterrupt from inside the step.
    monkeypatch.setattr(
        gym_manipulator, "step_env_and_process_transition", Mock(side_effect=KeyboardInterrupt)
    )
    cfg = Mock(mode="record" if record else None)
    cfg.env.fps = 1000
    cfg.dataset.push_to_hub = push
    cfg.dataset.num_episodes_to_record = 1
    if record:
        gym_manipulator.control_loop(Mock(), Mock(), Mock(), Mock(), cfg)
        dataset.clear_episode_buffer.assert_called_once()
        dataset.finalize.assert_called_once()
        assert dataset.push_to_hub.call_count == int(push)
    else:
        with pytest.raises(KeyboardInterrupt):
            gym_manipulator.control_loop(Mock(), Mock(), Mock(), Mock(), cfg)
