"""Serve OpenVLA LIBERO actions on a Linux GPU, independently of LeRobot's Transformers version."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
from PIL import Image
from remote_protocol import (
    ACTION_CONVENTION,
    MAX_BODY_BYTES,
    PROTOCOL_VERSION,
    SUITES,
    decode_image,
    openvla_to_libero_action,
    validate_action,
)

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-spatial"
LOGGER = logging.getLogger(__name__)


class OpenVLAPolicy:
    def __init__(self, model_id: str, suite: str, quantization: str, center_crop: bool):
        # This standalone example has its own pinned environment; these dependencies
        # must never be imported into the main LeRobot process.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        import tensorflow as tf
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        tf.config.set_visible_devices([], "GPU")  # TF only implements image preprocessing.
        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA server needs CUDA. Check the NVIDIA driver and CUDA-enabled PyTorch.")
        self.tf, self.torch = tf, torch
        self.suite, self.center_crop = suite, center_crop
        self.model_id = model_id
        self.dtype = torch.bfloat16
        options: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if quantization != "none":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
            )
            options["device_map"] = {"": 0}
        LOGGER.info("Loading %s (%s); the first run downloads the checkpoint", model_id, quantization)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, **options)
        if quantization == "none":
            self.model.to("cuda:0")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        stats = self.model.norm_stats
        self.unnorm_key = suite if suite in stats else f"{suite}_no_noops"
        if self.unnorm_key not in stats:
            raise ValueError(f"Checkpoint has no {suite} action statistics. Available: {list(stats)}")

    def preprocess(self, raw_image: np.ndarray) -> Image.Image:
        """Follow OpenVLA's LIBERO orientation, JPEG/Lanczos resize and 90%-area center crop."""
        tf = self.tf
        image = tf.convert_to_tensor(raw_image[::-1, ::-1].copy())
        image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
        image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
        if self.center_crop:
            image = tf.image.convert_image_dtype(image, tf.float32)
            offset = (1 - math.sqrt(0.9)) / 2
            image = tf.image.crop_and_resize(
                image[None], [[offset, offset, 1 - offset, 1 - offset]], [0], (224, 224)
            )[0]
            image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
        return Image.fromarray(image.numpy()).convert("RGB")

    def predict(self, image: np.ndarray, task: str) -> list[float]:
        prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
        inputs = self.processor(prompt, self.preprocess(image)).to("cuda:0", dtype=self.dtype)
        with self.torch.inference_mode():
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return openvla_to_libero_action(action)

    def metadata(self) -> dict:
        return {
            "backend": "openvla",
            "model": self.model_id,
            "suite": self.suite,
            "unnorm_key": self.unnorm_key,
        }


class MockPolicy:
    """Explicit transport/simulation diagnostic; never represents a model prediction."""

    def __init__(self, suite: str):
        self.suite = suite

    def predict(self, image: np.ndarray, task: str) -> list[float]:
        return [0, 0, 0, 0, 0, 0, -1]

    def metadata(self) -> dict:
        return {"backend": "mock", "model": "NO MODEL: transport test only", "suite": self.suite}


def make_server(policy: Any, host: str = "127.0.0.1", port: int = 8080) -> HTTPServer:
    # One simulator/client at a time. Serial request handling also avoids concurrent GPU calls.
    class Handler(BaseHTTPRequestHandler):
        timeout = 120

        def respond(self, status: int, value: dict) -> None:
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.respond(404, {"error": "Unknown route"})
                return
            self.respond(
                200,
                {"protocol": PROTOCOL_VERSION, "action_convention": ACTION_CONVENTION, **policy.metadata()},
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self.respond(404, {"error": "Unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("Request body is empty or too large")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or request.get("protocol") != PROTOCOL_VERSION:
                    raise ValueError("Incompatible request protocol")
                if request.get("suite") != policy.metadata()["suite"]:
                    raise ValueError("Requested LIBERO suite does not match loaded checkpoint")
                for key in ("request_id", "task"):
                    if not isinstance(request.get(key), str) or not 0 < len(request[key]) <= 4096:
                        raise ValueError(f"{key} must be a nonempty string of at most 4096 characters")
                image = decode_image(request.get("image_png"))
            except (ValueError, OSError) as error:
                self.respond(400, {"error": str(error)})
                return
            start = time.perf_counter()
            try:
                action = validate_action(policy.predict(image, request["task"]))
            except Exception:
                LOGGER.exception("Inference failed")
                self.respond(500, {"error": "Inference failed; see the GPU server log"})
                return
            self.respond(
                200,
                {
                    "protocol": PROTOCOL_VERSION,
                    "action_convention": ACTION_CONVENTION,
                    "request_id": request["request_id"],
                    "action": action,
                    "inference_ms": (time.perf_counter() - start) * 1000,
                },
            )

    return HTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--suite", choices=SUITES, default="libero_spatial")
    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"), default="4bit")
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--mock", action="store_true", help="No weights or GPU; diagnostic replies ONLY")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = (
        MockPolicy(args.suite)
        if args.mock
        else OpenVLAPolicy(args.model, args.suite, args.quantization, not args.no_center_crop)
    )
    with make_server(policy, args.host, args.port) as server:
        LOGGER.info("Ready at http://%s:%s — %s", args.host, args.port, policy.metadata())
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
