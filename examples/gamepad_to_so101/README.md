# SO-101 gamepad setup using the existing LeRobot script

This configuration runs `lerobot.rl.gym_manipulator` with Cartesian inverse
kinematics. It uses SDL's standardized input layout, verified with a Nintendo
Switch Pro Controller on macOS. It does not run a learned policy.

For day-to-day operation (controls, pre-run checklist, troubleshooting), see
[USAGE.md](./USAGE.md).

## Setup once

From the repository root, point `BASE_PYTHON` to an environment where LeRobot and
PyTorch already work:

```bash
BASE_PYTHON=/path/to/your/env/bin/python \
  bash examples/gamepad_to_so101/setup.sh
```

The setup creates `.venv-gamepad` inheriting that installation, installs pygame
and the pinned Placo dependencies there, and downloads the unmodified SO-101 URDF,
its meshes, and license from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101).
The downloaded commit is recorded in `outputs/gamepad/SO101/SOURCE.json` and reused
on subsequent setup runs. Setup does not connect to the follower.

## Launch

Edit `so101_gamepad.json` first: set `env.robot.port` to your follower's serial
port (`lerobot-find-port`) and `env.robot.id` to its saved calibration ID. The
config uses degree coordinates, no cameras, and no fixed reset pose.

From the repository root:

```bash
uv run --no-project --python .venv-gamepad/bin/python \
  python -m lerobot.rl.gym_manipulator \
  --config_path=examples/gamepad_to_so101/so101_gamepad.json
```

The configuration uses `mode: null`, so no dataset is recorded or uploaded.
Movement is expressed in the robot base frame, at up to 2 mm per axis per cycle
(15 Hz). The follower additionally limits joint targets to within 6 degrees of
their current position (6 percentage points for the gripper). Cartesian bounds
are geometric limits, not collision avoidance; check the surrounding workspace.

| Input | Behavior |
| --- | --- |
| Hold R (right shoulder) | Enable operator movement |
| Left stick up/down | X forward/back |
| Left stick left/right | Y left/right |
| Right stick up/down | Z up/down |
| ZL / ZR while holding R | Gripper decrease / increase |
| D-pad up / down while holding R | Tilt wrist up / down (`wrist_flex`) |
| D-pad left / right while holding R | Roll wrist (`wrist_roll`) |
| Release R | Return to the script's neutral action |
| Minus or Ctrl-C | Exit, disconnect, release motor torque |

Support the arm before exiting. Direction follows the robot base coordinates;
try small stick deflections first. SDL reports face buttons using its standardized
A/B/Y names, which can differ from the Nintendo printed labels. Those buttons
retain the existing episode failure/rerecord/success controls; use Minus to quit.

The original pygame and HID reader defaults remain available. This config selects
`env.teleop.use_sdl_controller=true` to avoid hard-coded raw button indices.

## Startup communication errors

A `Lock` or `Torque_Enable` write can fail if a status packet is lost. The config
sets `env.robot.num_torque_retries=2`: each startup torque/Lock write gets at most
three attempts, and persistent failures still stop startup. The entry point also
attempts to disconnect the robot and controller when environment creation fails.

If motor 5 (`wrist_roll`) still reports no status packet, close other robot-control
processes and check that motor's power and daisy-chain connectors before retrying.
Support the arm and turn off its power before reseating motor cables. SDL/OpenCV
duplicate-class warnings are separate from this serial communication exception.
