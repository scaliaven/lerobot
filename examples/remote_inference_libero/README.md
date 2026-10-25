# Step 0: local LIBERO simulation, remote OpenVLA inference

This example adds the missing bridge:

```text
Laptop: LIBERO camera observation → lossless PNG → SSH tunnel
Linux 4070 Ti: OpenVLA preprocessing → GPU prediction → LIBERO action
Laptop: validate reply → env.step(action) → browser preview / MP4 / action log
```

The default is **`openvla/openvla-7b-finetuned-libero-spatial`**, evaluated on
**`libero_spatial`**, task 0, one episode. It needs no new training or demonstration
dataset. This is a checkpoint already fine-tuned for LIBERO; it is not zero-shot
control of an SO-101. LIBERO uses a simulated Panda arm.

OpenVLA is not a policy type in this LeRobot v0.6.1 checkout. This example therefore
has a standalone HTTP server and a direct LIBERO client. It uses the same
`hf-libero` dependency as LeRobot, without altering the core policy registry,
`uv.lock`, or the existing physical-robot example. OpenVLA's legacy Transformers
version lives in a separate environment.

Vanilla OpenVLA predicts **one action per request**. The simulator pauses while
waiting, then advances exactly once. This checks the full observation/reply/action
loop; it is not the asynchronous action-chunk controller used on the physical arm.

## Machines and prerequisites

- **GPU server:** Linux x86_64, NVIDIA driver, RTX 4070 Ti, `uv` on PATH, and this
  example directory. `setup-server` creates a separate Python 3.10 environment.
  Default 4-bit bitsandbytes loading reduces the 7B model's memory use for a 12 GB
  4070 Ti. Actual memory/latency must be checked on your machine. No FlashAttention
  compilation is required. Expect a substantial first checkpoint download.
- **Laptop:** LIBERO is supported on Linux. On **macOS**, use the supplied Linux
  container with CPU OSMesa rendering; no local model or GPU is required.
- **Docker Desktop on macOS:** start Docker, use version **4.34+**, and enable
  **Settings → Resources → Network → Enable host networking**, then apply/restart.
  The launcher uses this to reach the SSH tunnel bound to localhost. The container
  targets `linux/amd64`; Apple Silicon runs it through emulation, so rendering may
  be slow. Allocate enough Docker memory for PyTorch, LIBERO and its assets.
- Both machines need the example files. A checkout of LeRobot is convenient, but
  the two standalone Python processes do not import the LeRobot package itself.

[Install uv](https://docs.astral.sh/uv/getting-started/installation/).
[Docker host networking](https://docs.docker.com/engine/network/drivers/host/).
[LeRobot's Linux requirement for LIBERO](https://huggingface.co/docs/lerobot/v0.6.1/libero).

## 1. Configure both machines

Run commands from the repository root. On each machine:

```bash
cp examples/remote_inference_libero/.env.example examples/remote_inference_libero/.env
```

On the laptop, set `REMOTE_SSH` to your working SSH alias or `user@hostname`.
Configure custom SSH keys/ports/jump hosts in `~/.ssh/config`. Both `.env` copies
must agree on `SERVER_PORT` and `LIBERO_SUITE`. These files contain Bash code; use
only trusted configurations. Keep credentials in your usual SSH/Hugging Face
credential stores rather than in `.env`.

The defaults select a matching model and suite, so you do not need a model ID of
your own. If switching suites, select a checkpoint with statistics for that suite
and set its episode limit; the server/client reject a suite mismatch.

## 2. GPU computer: install and start OpenVLA

```bash
bash examples/remote_inference_libero/run.sh setup-server
bash examples/remote_inference_libero/run.sh server
```

Wait for the **Ready** message. The first start downloads weights and the model's
custom Hugging Face code (`trust_remote_code=True`, as required by OpenVLA).
The server binds to `127.0.0.1:8080` by default and loads the checkpoint once.
Stop the previous SO-101 policy server first if it is using that port.

## 3. Laptop: open the tunnel

```bash
bash examples/remote_inference_libero/run.sh tunnel
```

Leave it running. In a second laptop terminal:

```bash
bash examples/remote_inference_libero/run.sh health
```

The JSON should identify `backend: openvla`, the model ID, and `libero_spatial`.
This confirms connectivity and loading, but does not yet test a prediction.

## 4. Mac laptop: start the simulator

```bash
bash examples/remote_inference_libero/run.sh build-client
bash examples/remote_inference_libero/run.sh docker-client
```

Open **[http://127.0.0.1:8090](http://127.0.0.1:8090)** while the episode is running.
It shows the simulator frame, task, applied action and round-trip/inference timing.
The container may download LIBERO assets on its first run. GPU model weights are
downloaded only on the remote machine. The Docker volume `lerobot-libero-cache`
keeps the simulation assets between runs.

Outputs are written to a new `examples/remote_inference_libero/outputs/<timestamp>/`
directory on your laptop: `episode-000.mp4`, per-step `episode-000.jsonl`,
`summary.json`, and the final preview image/status. MP4 playback uses simulation
time (20 fps); wall-clock execution can be much slower. The live preview stops
when the client exits, and the saved artifacts remain available.

Ctrl-C stops a foreground process. There is no physical robot connection in this
example. An inference error or timeout stops the rollout; no fallback action is
applied and no request is automatically retried.

### Native Linux laptop instead

Install the OSMesa/OpenGL runtime libraries (`libosmesa6`, `libgl1`, `libegl1`,
`libglib2.0-0`, `libglfw3`) and build tools (`g++`, `cmake`) using your distribution's
package manager, then:

```bash
bash examples/remote_inference_libero/run.sh setup-client
bash examples/remote_inference_libero/run.sh client
```

The native client defaults to `MUJOCO_GL=osmesa`. Set `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl` if you prefer a working GPU renderer. `--list-tasks`,
`--no-video`, `--max-steps` and other options are available through the Python
client's `--help`. The wrapper exposes the common settings in `.env`.

## Validation and model conventions

- Requests include the suite and an observation ID; replies must match that ID
  and the versioned seven-dimensional LIBERO delta-action contract.
- Raw `agentview_image` frames go over the wire unchanged. The GPU server rotates
  them 180°, applies the OpenVLA JPEG/Lanczos resize to 224×224, then applies its
  90%-area center crop. It uses the checkpoint's suite-specific unnormalization.
- The server converts the predicted gripper from `[0, 1]` to LIBERO's
  `-1=open, +1=closed` convention. The client does not convert it again.
- Episodes start from LIBERO's fixed initial states, take 10 settling steps, and
  use a newly constructed simulator per trial. Success comes from LIBERO itself.
- The HTTP endpoint is intended for your SSH tunnel; it is not an authenticated
  public model-serving endpoint.
- The preview identifies the backend. `openvla_server.py --mock` is an explicitly
  labeled no-model diagnostic for transport tests, never a default or fallback.

The CPU tests exercise real loopback HTTP requests with a deterministic backend,
PNG orientation, action/gripper validation, suite mismatch, stale replies, error
handling, and applying returned actions to a deterministic environment interface.
They do **not** establish LIBERO task success or real GPU performance.

```bash
uv run --no-project --python 3.10 --with numpy==1.26.4 --with Pillow --with pytest \
  python -m pytest examples/remote_inference_libero/test_remote_inference.py -q
```

Upstream references:
[OpenVLA LIBERO checkpoint](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial),
[evaluation loop](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py),
[image preprocessing](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py),
[model preprocessing](https://github.com/openvla/openvla/blob/main/experiments/robot/openvla_utils.py),
[gripper conventions](https://github.com/openvla/openvla/blob/main/experiments/robot/robot_utils.py).
