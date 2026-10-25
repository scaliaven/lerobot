# Keyboard control for an SO-101 follower

Run from the LeRobot repository, using the same port and ID as calibration:

```bash
uv run python examples/keyboard_to_so101/teleoperate.py \
  --port=/dev/tty.usbmodemXXXXXXXX \
  --id=my_awesome_follower_arm
```

Find the port with `lerobot-find-port`. To reuse an existing environment instead of
the repository's `.venv`, run the same script with that environment's `python`.

Keep the terminal focused. This uses LeRobot's `TerminalKeyListener`, so it needs
an interactive macOS/Linux terminal and no global keyboard capture permissions.
Each key event sends one small joint target; holding a key produces the terminal's
normal repeated key events. This is joint control, with no Cartesian/IK solver.

| Keys | Joint | Change per event |
| --- | --- | --- |
| A / D | Base rotation (`shoulder_pan`) | +1 / -1 degree |
| W / S | Shoulder lift | +1 / -1 degree |
| R / F | Elbow bend | +1 / -1 degree |
| T / G | Wrist bend | +1 / -1 degree |
| Y / H | Wrist roll | +1 / -1 degree |
| O / P | Gripper | +2 / -2 percentage points |
| Q, Esc, Ctrl-C | Exit | Disconnect and release torque |

Joint directions follow the calibrated motor coordinates. O/P are numeric gripper
increments; observe which direction opens your particular gripper. Use `--step=0.5`
or `--gripper-step=1` for smaller changes (maximum accepted step: 5).
Support the arm before exiting because disconnect releases torque.

The example checks saved calibration, clips targets to calibrated ranges, bases
each target on a fresh position reading, drops old key events, and restores the
terminal and disconnects on errors. It does not connect cameras or record data.

## Existing LeRobot input support

- `src/lerobot/teleoperators/keyboard/teleop_keyboard.py`: generic keyboard input,
  arrow/Shift/Ctrl Cartesian commands via `KeyboardEndEffectorTeleop`, and WASD
  rover commands via `KeyboardRoverTeleop`.
- `src/lerobot/teleoperators/gamepad/teleop_gamepad.py`: `GamepadTeleop` produces
  `delta_x`, `delta_y`, `delta_z`, and gripper commands. Dependencies are provided
  by the `gamepad` extra. macOS also has `hidapi_fallback=true` for supported HID
  controllers when pygame input does not work.
- `docs/source/hilserl.mdx`: the existing gamepad/keyboard arm workflow uses a
  processor pipeline with inverse kinematics and a robot URDF.

`lerobot-teleoperate --teleop.type=gamepad --robot.type=so101_follower ...`
does not install that IK pipeline: its default processors pass through actions,
while the follower expects joint `.pos` targets. This example supplies the missing
direct keyboard-to-joint mapping; it does not change the gamepad implementation.
