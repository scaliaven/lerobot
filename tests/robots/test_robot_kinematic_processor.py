#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import pytest

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEAction,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
    InverseKinematicsRLStep,
    WristJogStep,
)

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EE_KEYS = {f"ee.{k}" for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]}


def _joint_bucket(feature_type: FeatureType) -> dict[str, PolicyFeature]:
    return {f"{n}.pos": PolicyFeature(type=feature_type, shape=(1,)) for n in MOTOR_NAMES}


@pytest.mark.parametrize(
    ("step_cls", "bucket", "feature_type"),
    [
        (ForwardKinematicsJointsToEEAction, PipelineFeatureType.ACTION, FeatureType.ACTION),
        (ForwardKinematicsJointsToEEObservation, PipelineFeatureType.OBSERVATION, FeatureType.STATE),
    ],
)
def test_fk_feature_schema(step_cls, bucket, feature_type):
    features = {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}
    features[bucket] = _joint_bucket(feature_type)
    out = step_cls(kinematics=None, motor_names=MOTOR_NAMES).transform_features(features)[bucket]
    assert set(out) == EE_KEYS
    assert {feature.type for feature in out.values()} == {feature_type}


EE_ACTION = dict.fromkeys(EE_KEYS, 0.0)


@pytest.mark.parametrize(
    ("step", "action"),
    [
        (
            EEReferenceAndDelta(kinematics=None, end_effector_step_sizes={}, motor_names=MOTOR_NAMES),
            dict(EE_ACTION),
        ),
        (InverseKinematicsEEToJoints(kinematics=None, motor_names=MOTOR_NAMES), dict(EE_ACTION)),
        (GripperVelocityToJoint(), {**EE_ACTION, "ee.gripper_vel": 0.0}),
        (InverseKinematicsRLStep(kinematics=None, motor_names=MOTOR_NAMES), dict(EE_ACTION)),
    ],
    ids=[
        "ee_reference_and_delta",
        "inverse_kinematics_ee_to_joints",
        "gripper_velocity_to_joint",
        "inverse_kinematics_rl_step",
    ],
)
def test_missing_observation_raises_value_error(step, action):
    """A transition without an observation must surface the documented ValueError.

    `RobotProcessorPipeline.process_action` builds its transition with
    `create_transition(action=...)`, which sets `TransitionKey.OBSERVATION` to None.
    These steps used to call `.copy()` on that before the None check, so the guard
    below them was unreachable and an AttributeError escaped instead.
    """
    transition = create_transition(action=action)

    with pytest.raises(ValueError, match="Joints observation"):
        step(transition)


def _wrist_jog_step(deltas, bounds=None, fk=None, max_relative_target=None):
    kinematics = Mock()
    kinematics.robot.get_joint_limits.side_effect = lambda name: np.deg2rad([-95.0, 95.0])
    kinematics.forward_kinematics.side_effect = fk or (lambda q: np.diag([1.0, 1.0, 1.0, 1.0]) * q[3])
    ee_reference = EEReferenceAndDelta(
        kinematics=kinematics, end_effector_step_sizes={}, motor_names=MOTOR_NAMES
    )
    ik = InverseKinematicsRLStep(kinematics=kinematics, motor_names=MOTOR_NAMES)
    step = WristJogStep(
        kinematics=kinematics,
        motor_names=MOTOR_NAMES,
        ee_reference_step=ee_reference,
        ik_step=ik,
        get_wrist_deltas=lambda: deltas,
        end_effector_bounds=bounds or {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        max_relative_target=max_relative_target,
    )
    return step, ee_reference, ik


def _ik_transition(wrist_flex=0.0, wrist_roll=0.0, measured=None):
    q = np.array([0.0, -40.0, 40.0, wrist_flex, wrist_roll, 10.0])
    action = {f"{name}.pos": value for name, value in zip(MOTOR_NAMES, q, strict=True)}
    observation = dict(action) if measured is None else measured
    return create_transition(observation=observation, action=action, complementary_data={"IK_solution": q})


def test_wrist_jog_without_input_leaves_ik_command_unchanged():
    step, ee_reference, ik = _wrist_jog_step((0, 0))
    transition = _ik_transition()
    assert step(transition) is transition
    assert ik.q_curr is None and ee_reference._command_when_disabled is None


def test_wrist_jog_moves_wrist_and_updates_ik_state():
    step, ee_reference, ik = _wrist_jog_step((-1, 1))
    out = step(_ik_transition())
    action = out[TransitionKey.ACTION]
    assert action["wrist_flex.pos"] == -2.0
    assert action["wrist_roll.pos"] == 2.0
    assert action["elbow_flex.pos"] == 40.0
    np.testing.assert_allclose(out[TransitionKey.COMPLEMENTARY_DATA]["IK_solution"][3:5], [-2.0, 2.0])
    np.testing.assert_allclose(ik.q_curr[3:5], [-2.0, 2.0])
    np.testing.assert_allclose(ee_reference._command_when_disabled, np.eye(4) * -2.0)


def test_wrist_jog_stops_at_urdf_joint_limit():
    step, _, _ = _wrist_jog_step((1, 0))
    out = step(_ik_transition(wrist_flex=94.0))
    assert out[TransitionKey.ACTION]["wrist_flex.pos"] == pytest.approx(95.0)


def test_wrist_jog_target_stays_near_measured_position():
    # A blocked wrist reads 0 while the IK target has already run ahead to 30.
    step, _, _ = _wrist_jog_step((1, 0))
    measured = {f"{name}.pos": 0.0 for name in MOTOR_NAMES}
    out = step(_ik_transition(wrist_flex=30.0, measured=measured))
    assert out[TransitionKey.ACTION]["wrist_flex.pos"] == pytest.approx(6.0)


def test_wrist_jog_that_leaves_workspace_is_dropped():
    def fk(q):
        pose = np.eye(4)
        pose[2, 3] = -0.099 - 0.002 * q[3]  # Positive wrist_flex lowers the tip.
        return pose

    bounds = {"min": [-1.0, -1.0, -0.1], "max": [1.0, 1.0, 1.0]}
    step, _, ik = _wrist_jog_step((1, 0), bounds=bounds, fk=fk)
    transition = _ik_transition()
    out = step(transition)
    assert out[TransitionKey.ACTION]["wrist_flex.pos"] == 0.0

    step, _, _ = _wrist_jog_step((-1, 0), bounds=bounds, fk=fk)
    assert step(transition)[TransitionKey.ACTION]["wrist_flex.pos"] == -2.0


@pytest.mark.parametrize("max_relative_target, accepted", [(None, True), (6.0, False)])
def test_wrist_jog_checks_the_command_the_follower_sends(max_relative_target, accepted):
    def fk(q):
        pose = np.eye(4)
        pose[0, 3] = 0.001 * (q[3] - q[1])
        return pose

    # The IK target (shoulder 10, wrist 8 -> 10) stays in bounds, but the shoulder lags at 0 and the
    # follower caps it to 6, so the command it sends is already outside and the jog pushes it further.
    bounds = {"min": [-1.0, -1.0, -1.0], "max": [0.0, 1.0, 1.0]}
    step, _, _ = _wrist_jog_step((1, 0), bounds=bounds, fk=fk, max_relative_target=max_relative_target)
    transition = _ik_transition(wrist_flex=8.0)
    transition[TransitionKey.COMPLEMENTARY_DATA]["IK_solution"][1] = 10.0
    transition[TransitionKey.OBSERVATION]["shoulder_lift.pos"] = 0.0
    out = step(transition)
    assert bool(out[TransitionKey.ACTION]["wrist_flex.pos"] == 10.0) == accepted
