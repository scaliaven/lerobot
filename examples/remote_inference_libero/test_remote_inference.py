"""CPU tests for the real HTTP bridge and rollout loop; no checkpoints or LIBERO assets required."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest
from libero_client import Artifacts, run_episode
from openvla_server import MockPolicy, make_server
from PIL import Image
from remote_protocol import (
    ACTION_CONVENTION,
    MAX_BODY_BYTES,
    RemotePolicy,
    decode_image,
    encode_image,
    json_request,
    openvla_to_libero_action,
    validate_action,
)


@contextmanager
def running_server(policy):
    with make_server(policy, port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.fixture
def image():
    # Asymmetric corners reveal accidental orientation changes in the transport.
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    image[-1, -1] = [0, 255, 0]
    return image


def test_raw_image_transport_preserves_orientation(image):
    np.testing.assert_array_equal(decode_image(encode_image(image)), image)


@pytest.mark.parametrize("value", ["not base64", "", 3])
def test_invalid_image_rejected(value):
    with pytest.raises(ValueError):
        decode_image(value)


@pytest.mark.parametrize(
    "action",
    [[0] * 6, [0] * 8, [0] * 6 + [float("nan")], [0] * 6 + [float("inf")], [2] + [0] * 6, [True] * 7],
)
def test_invalid_action_rejected(action):
    with pytest.raises(ValueError):
        validate_action(action)


@pytest.mark.parametrize(
    "gripper,expected", [(-0.007843, 1), (0, 1), (0.25, 1), (0.5, 0), (0.75, -1), (1, -1), (1.007843, -1)]
)
def test_gripper_conversion_matches_openvla_libero(gripper, expected):
    raw = [0.1, -0.2, 0.3, 0, 0, 0, gripper]
    action = openvla_to_libero_action(raw)
    assert action[:6] == raw[:6]
    assert action[-1] == expected
    assert raw[-1] == gripper  # Conversion does not mutate the model's output.


def test_actual_http_round_trip(image):
    class RecordingPolicy(MockPolicy):
        def predict(self, received, task):
            np.testing.assert_array_equal(received, image)
            assert task == "put the bowl on the plate"
            return [0.1, 0, 0, 0, 0, 0, -1]

    with running_server(RecordingPolicy("libero_spatial")) as endpoint:
        client = RemotePolicy(endpoint, "libero_spatial")
        assert client.check()["backend"] == "mock"
        action, reply = client.predict(image, "put the bowl on the plate", "episode0:step0")
        assert action == [0.1, 0, 0, 0, 0, 0, -1]
        assert reply["request_id"] == "episode0:step0"
        assert reply["action_convention"] == ACTION_CONVENTION
        assert reply["inference_ms"] >= 0


def test_suite_mismatch_stops_before_inference():
    with (
        running_server(MockPolicy("libero_spatial")) as endpoint,
        pytest.raises(ValueError, match="does not match"),
    ):
        RemotePolicy(endpoint, "libero_goal").check()


def test_invalid_request_returns_useful_error():
    with running_server(MockPolicy("libero_spatial")) as endpoint:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            json_request(endpoint, "/predict", {"protocol": 42})
        # Send just the oversized header: the server rejects it before reading a body.
        # Sending a whole oversized body races TCP's close and can produce a reset.
        address = urlsplit(endpoint)
        connection = HTTPConnection(address.hostname, address.port, timeout=5)
        try:
            connection.putrequest("POST", "/predict")
            connection.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
            connection.endheaders()
            reply = connection.getresponse()
            assert reply.status == 400
            assert "too large" in reply.read().decode()
        finally:
            connection.close()


def test_inference_failure_is_not_replaced_with_fake_action(image):
    class BrokenPolicy(MockPolicy):
        def predict(self, image, task):
            raise RuntimeError("GPU unavailable")

    with (
        running_server(BrokenPolicy("libero_spatial")) as endpoint,
        pytest.raises(RuntimeError, match="HTTP 500"),
    ):
        RemotePolicy(endpoint, "libero_spatial").predict(image, "test", "one")


def test_mismatched_reply_is_rejected(image, monkeypatch):
    monkeypatch.setattr(
        "remote_protocol.json_request",
        lambda *a, **kw: {
            "protocol": 1,
            "action_convention": ACTION_CONVENTION,
            "request_id": "stale",
            "action": [0] * 7,
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        RemotePolicy("http://127.0.0.1:1", "libero_spatial").predict(image, "test", "current")


class FakeEnvironment:
    """Deterministic simulation interface, deliberately separate from the HTTP model backend."""

    def __init__(self, image):
        self.image = image
        self.actions = []

    def seed(self, seed):
        assert seed == 0

    def reset(self):
        self.actions.clear()

    def set_init_state(self, state):
        assert state == "initial"
        return {"agentview_image": self.image}

    def step(self, action):
        self.actions.append(action)
        return {"agentview_image": self.image}, 0.0, False, {}

    def check_success(self):
        return len(self.actions) >= 4


def test_simulation_loop_applies_actual_http_replies_and_logs(tmp_path: Path, image):
    class MovingPolicy(MockPolicy):
        def predict(self, image, task):
            return [0.2, 0, 0, 0, 0, 0, -1]

    env = FakeEnvironment(image)
    with running_server(MovingPolicy("libero_spatial")) as endpoint:
        policy = RemotePolicy(endpoint, "libero_spatial")
        summary = run_episode(
            env, "initial", policy, "test task", 0, 10, 2, Artifacts(tmp_path), policy.check()
        )
    assert summary["success"]
    assert summary["steps"] == 2
    assert env.actions[:2] == [[0, 0, 0, 0, 0, 0, -1]] * 2  # Settle steps.
    assert env.actions[2:] == [[0.2, 0, 0, 0, 0, 0, -1]] * 2
    records = [json.loads(line) for line in (tmp_path / "episode-000.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[-1]["success"]
    assert records[0]["request_id"] != records[1]["request_id"]
    assert records[0]["round_trip_ms"] >= 0
    assert records[0]["backend"] == "mock"
    np.testing.assert_array_equal(np.array(Image.open(tmp_path / "latest.png")), image[::-1, ::-1])


def test_failed_reply_does_not_advance_simulation(tmp_path: Path, image):
    class BrokenClient:
        def predict(self, *args):
            raise RuntimeError("Tunnel disconnected")

    env = FakeEnvironment(image)
    with pytest.raises(RuntimeError, match="disconnected"):
        run_episode(env, "initial", BrokenClient(), "test", 0, 10, 0, Artifacts(tmp_path), {})
    assert env.actions == []
