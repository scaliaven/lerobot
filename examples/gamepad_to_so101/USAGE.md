# Driving the SO-101 with a gamepad

How to run and operate `so101_gamepad.json` day to day. For one-time setup
(`.venv-gamepad`, URDF download) see [README.md](./README.md).

Tested with a Nintendo Switch Pro Controller on macOS, read through SDL's
standardized controller layout (`env.teleop.use_sdl_controller=true`).

## Before you start

1. The follower is powered, calibrated, and on the port set in `so101_gamepad.json`.
2. The controller is paired over Bluetooth or plugged in by USB.
3. **Lift the shoulder a few degrees off its folded rest by hand.** The SO-101
   URDF limits `shoulder_lift` to ±100°, and the folded rest reads about −104°.
   Starting there makes the IK keep pulling the shoulder toward −100°, which
   shows up as repeated `Relative goal position magnitude had to be clamped`
   warnings.
4. Clear the workspace. The Cartesian bounds in the config are geometric limits,
   not collision avoidance.

## Launch

From the repository root:

```bash
.venv-gamepad/bin/python -m lerobot.rl.gym_manipulator \
  --config_path=examples/gamepad_to_so101/so101_gamepad.json
```

Keep the trailing `\`. If the command wraps when pasted without it, the first
line runs alone and fails with `Missing required field(s) env, dataset`.

Wait for `Starting control loop at 15 FPS` before touching the controller.

## Controls

**Nothing moves unless R is held.** This includes the gripper. Releasing R
holds the last commanded pose.

```text
        ZL (open gripper)                 ZR (close gripper)
        L                                 R  (hold to enable)
   ┌──────────────────────────────────────────────┐
   │   Left stick                    Right stick   │
   │   ↑/↓ forward / back            ↑/↓ up / down │
   │   ←/→ left / right                            │
   │   D-pad ↑/↓ tilt wrist   ←/→ roll wrist       │
   │            [−] quit                           │
   └──────────────────────────────────────────────┘
```

| Input | Effect |
| --- | --- |
| Hold **R** (right shoulder button, under the index finger) | Enable movement |
| Left stick up / down | Move forward / back (X) |
| Left stick left / right | Move left / right (Y) |
| **Right** stick up / down | Move up / down (Z) |
| ZL (trigger behind L), with R held | Open gripper |
| ZR (trigger behind R), with R held | Close gripper |
| D-pad up / down, with R held | Tilt the gripper up / down (`wrist_flex`, 30°/s) |
| D-pad left / right, with R held | Roll the gripper (`wrist_roll`, 30°/s) |
| Minus (−) or Ctrl-C | Quit: disconnects and releases torque |

The sticks move the gripper while the IK keeps its current tilt. The D-pad turns
the wrist joints directly; the new tilt is then what the sticks hold. Large up/down
stick moves can still change the tilt somewhat, because the 5-joint arm cannot
hold every orientation. Tilting the wrist also moves the gripper tip a few
centimetres, since the tip sits about 10 cm past the wrist joint.

Face buttons keep the episode controls from the HIL-SERL workflow (SDL names:
Y = success, A = failure, B = re-record). They are not needed for plain
teleoperation; this config does not record anything (`mode: null`).

Motion is in the robot base frame, up to 2 mm per axis per cycle at 15 Hz
(about 3 cm/s). The follower also limits each joint target to 6° from its
current position (6 percentage points for the gripper). Start with small stick
deflections.

**Support the arm before quitting.** Exiting releases motor torque and the arm
drops.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Sticks do nothing | R is not held. Hold R first, then move the stick. |
| "Can't move up" | Up/down is the **right** stick; left stick up is forward. If the right stick with R held still doesn't lift the arm, gravity sag may be absorbing the 2 mm step: raise `end_effector_step_sizes` in the config (e.g. to `0.005`). |
| Gripper doesn't open/close | Hold R while pressing ZL/ZR. |
| Arm creeps while R is held, sticks untouched | Stick drift. The SDL reader ignores deflections below 0.25. Recalibrate the sticks in the controller's system settings if drift exceeds that. |
| No input at all, even with R held | macOS delivers controller input only to the foreground app. The SDL reader sets `SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1` to allow it while Terminal is in front. Close other apps that may hold the controller (Steam, emulators) and reconnect it. |
| `No SDL-compatible gamepad detected` | The controller isn't connected. Re-pair it, or plug it in by USB. |
| `Relative goal position magnitude had to be clamped` repeating while idle | A joint started outside the URDF limits; see step 3 of *Before you start*. Occasional warnings during fast motion are expected. |
| `objc[...] Class ... is implemented in both ...` at startup | Harmless here. OpenCV and PyAV each bundle FFmpeg's capture library, and OpenCV and pygame each bundle SDL. This config uses no cameras. |
| `pkg_resources is deprecated` | Harmless warning from pygame. |
| `There is no status packet` / `Torque_Enable` error at startup | See *Startup communication errors* in [README.md](./README.md). |

## Checking the controller without the robot

This prints what the reader sees for 15 seconds and does not open the serial
port:

```bash
.venv-gamepad/bin/python - <<'PY'
import time
from lerobot.teleoperators.gamepad.gamepad_utils import SDLGamepadController

pad = SDLGamepadController()
pad.start()
print("Connected:", pad.controller.name)
end = time.time() + 15
try:
    while time.time() < end:
        pad.update()
        dx, dy, dz = pad.get_deltas()
        print(f"R={pad.should_intervene()!s:5} dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f} "
              f"gripper={pad.gripper_command():5}", end="\r", flush=True)
        time.sleep(0.05)
finally:
    print()
    pad.stop()
PY
```

Holding R should show `R=True`; the sticks should move `dx`, `dy`, `dz` toward
±1.00; ZL and ZR should show `open` and `close`.
