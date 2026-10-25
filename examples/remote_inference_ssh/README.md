# SO-101 remote inference over SSH (LeRobot v0.6.1)

Run model inference on a Linux GPU computer and keep USB motor/camera access on
the computer beside the robot. This wrapper uses the existing `async_inference`
server/client in v0.6.1. It does not use the newer development-branch remote API.

For the simulation-first check (LIBERO on the laptop, OpenVLA on the GPU), use
[`../remote_inference_libero/README.md`](../remote_inference_libero/README.md).

## Prerequisites

- Both machines: this v0.6.1 checkout (including this example), Bash, and `uv` on PATH.
  Install uv using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
  If cloning separately, select tag `v0.6.1` and copy this example directory into it.
- GPU machine: a working NVIDIA driver; `nvidia-smi` should show the 4070 Ti.
- Robot computer: a calibrated SO-101 with working cameras and its existing calibration ID.
- SSH access from the robot computer to the GPU machine. An existing SSH config
  alias works, including its configured key, port, and jump host.
- A compatible trained checkpoint. GPU inference does not make a base model a
  zero-shot controller. Match the embodiment, calibration convention, camera names,
  image sizes, task, control frequency, and action chunk size to the checkpoint.

No dataset is needed just to run inference. Model weights are loaded on the GPU
machine when the client connects. The client still installs PyTorch for data and
action handling, but does not load or execute the policy model.

## Configure

From the repository root, on each machine:

```bash
cp examples/remote_inference_ssh/.env.example examples/remote_inference_ssh/.env
```

Edit `.env`. The GPU machine only needs `POLICY_TYPE`, `SERVER_PORT` and `FPS`.
The robot computer also needs the SSH destination, model, task, USB port,
calibration ID, camera configuration and action chunk size. `MODEL` is resolved
on the GPU machine. For a private Hub model, authenticate there using a Hugging
Face token through your usual credential setup.

The `.env` file is ignored by Git and is executed as Bash, so use a trusted file.
The two copies must agree on `SERVER_PORT`, `FPS` and `POLICY_TYPE`.

## Start

On the **GPU machine**, install once, then start the server:

```bash
bash examples/remote_inference_ssh/run.sh setup-server
bash examples/remote_inference_ssh/run.sh server
```

On the **robot computer**, install once, then open the tunnel and leave it running:

```bash
bash examples/remote_inference_ssh/run.sh setup-client
bash examples/remote_inference_ssh/run.sh tunnel
```

In another terminal on the **robot computer**, inspect the client command first:

```bash
bash examples/remote_inference_ssh/run.sh --dry-run client
```

When your checkpoint and robot configuration are ready, this command connects the
robot and begins executing model actions:

```bash
bash examples/remote_inference_ssh/run.sh client
```

Ctrl-C stops each foreground command. Keep a physical way to stop/power off the arm
within reach during evaluation. This wrapper adds no disconnect watchdog to the
underlying async client; do not treat a tunnel failure as an emergency stop.

## Troubleshooting

- Run `setup-server` again after changing policy families. Install any additional
  extras your checkpoint needs. Setup smoke-checks imports but does not download weights.
- Both setup commands synchronize the repository's `.venv` using `uv.lock`;
  runtime commands use `--no-sync` to preserve those installed extras.
- The policy server intentionally binds only `127.0.0.1`. Keep its pickle-based
  gRPC endpoint inside the SSH tunnel rather than exposing port 8080 publicly.
- If local port 18080 is occupied, change `LOCAL_PORT` on the robot computer.
- If the action queue empties, inspect inference time and network bandwidth/latency.
  Tune `ACTIONS_PER_CHUNK` and `CHUNK_SIZE_THRESHOLD` within your model's limits.
  Keep both machines' clocks synchronized for meaningful latency logs.
- Camera indices and robot serial paths belong to the robot computer, never the GPU server.
- The scripts assume Linux/macOS shell environments; native Windows is not covered.

Reference: [LeRobot async inference](https://huggingface.co/docs/lerobot/v0.6.1/async).
