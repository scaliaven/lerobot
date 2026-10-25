"""Small, versioned JSON/PNG protocol for simulation-only remote OpenVLA inference."""

from __future__ import annotations

import base64
import io
import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

PROTOCOL_VERSION = 1
ACTION_CONVENTION = "libero_osc_pose_delta"
MAX_BODY_BYTES = 2 * 1024 * 1024
IMAGE_SIZE = 256
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def validate_action(value: object) -> list[float]:
    """Reject malformed replies before passing anything to the simulator."""
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError("Expected one seven-dimensional LIBERO action")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError("Action components must be numbers")
    action = [float(x) for x in value]
    if any(not math.isfinite(x) or abs(x) > 1 for x in action):
        raise ValueError("LIBERO action components must be finite and within [-1, 1]")
    return action


def openvla_to_libero_action(value: object) -> list[float]:
    """Convert OpenVLA's [0, 1] gripper to LIBERO's -1=open, +1=closed convention."""
    action = np.array(value, dtype=np.float64, copy=True)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("OpenVLA must return seven finite action components")
    # OpenVLA's decoder skips unnormalization for the gripper, so bin centers can fall just outside [0, 1].
    gripper = np.clip(action[-1], 0, 1)
    action[-1] = -np.sign(2 * gripper - 1)
    return validate_action(action.tolist())


def encode_image(image: np.ndarray) -> str:
    """Send the raw agentview image; orientation and model preprocessing happen once on the server."""
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"Expected raw RGB uint8 image with shape ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)")
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode_image(value: object) -> np.ndarray:
    if not isinstance(value, str) or len(value) > MAX_BODY_BYTES:
        raise ValueError("image_png must be a base64-encoded PNG")
    try:
        data = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Expected an RGB {IMAGE_SIZE}x{IMAGE_SIZE} PNG")
            return np.array(image)
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid image_png: {error}") from error


def json_request(endpoint: str, route: str, body: dict | None = None, timeout: float = 120) -> dict:
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    request = Request(endpoint.rstrip("/") + route, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: explicit HTTP endpoint
            raw = response.read(MAX_BODY_BYTES + 1)
    except HTTPError as error:
        detail = error.read(4096).decode(errors="replace")
        raise RuntimeError(f"Policy server HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Policy server unreachable at {endpoint}: {error}") from error
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError("Policy server reply is too large")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Policy server reply must be a JSON object")
    return result


class RemotePolicy:
    def __init__(self, endpoint: str, suite: str, timeout: float = 120):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("Endpoint must be an http:// or https:// URL")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        self.endpoint, self.suite, self.timeout = endpoint, suite, timeout

    def check(self) -> dict:
        info = json_request(self.endpoint, "/health", timeout=self.timeout)
        if info.get("protocol") != PROTOCOL_VERSION or info.get("action_convention") != ACTION_CONVENTION:
            raise ValueError("Incompatible policy server protocol or action convention")
        if info.get("suite") != self.suite:
            raise ValueError(f"Client suite {self.suite!r} does not match server suite {info.get('suite')!r}")
        return info

    def predict(self, image: np.ndarray, task: str, request_id: str) -> tuple[list[float], dict]:
        reply = json_request(
            self.endpoint,
            "/predict",
            {
                "protocol": PROTOCOL_VERSION,
                "suite": self.suite,
                "request_id": request_id,
                "task": task,
                "image_png": encode_image(image),
            },
            timeout=self.timeout,
        )
        if reply.get("request_id") != request_id:
            raise ValueError("Policy reply does not match the current observation")
        if reply.get("protocol") != PROTOCOL_VERSION or reply.get("action_convention") != ACTION_CONVENTION:
            raise ValueError("Incompatible action reply")
        return validate_action(reply.get("action")), reply
