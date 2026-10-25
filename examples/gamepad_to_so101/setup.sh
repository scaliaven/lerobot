#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

# Reuse the existing LeRobot installation without modifying its environment.
BASE_PYTHON="${BASE_PYTHON:-python3.12}"
uv run --no-project --python "$BASE_PYTHON" python -c 'import lerobot, torch'
if [[ ! -x .venv-gamepad/bin/python ]]; then
    uv venv --python "$BASE_PYTHON" --system-site-packages .venv-gamepad
fi
uv pip install --python .venv-gamepad/bin/python \
    'pygame>=2.6,<3' 'placo>=0.9.6,<0.9.16' \
    'cmeel-urdfdom>=4,<5' 'cmeel-tinyxml2<11'

uv run --no-project --python .venv-gamepad/bin/python python - <<'PY'
import json
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

destination = Path("outputs/gamepad/SO101")
source_record = destination / "SOURCE.json"

def download(url):
    return subprocess.check_output([
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--connect-timeout", "10", "--max-time", "60", "--retry", "2", url,
    ])

if source_record.exists():
    revision = json.loads(source_record.read_text())["revision"]
else:
    url = "https://api.github.com/repos/TheRobotStudio/SO-ARM100/commits/main"
    revision = json.loads(download(url))["sha"]
base = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{revision}/"

def fetch(relative, source=None):
    target = destination / relative
    if target.exists():
        return target
    data = download(base + (source or f"Simulation/SO101/{relative}"))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target

urdf = fetch("so101_new_calib.urdf")
meshes = {mesh.attrib["filename"] for mesh in ET.parse(urdf).findall(".//mesh")}
for mesh in meshes:
    if Path(mesh).is_absolute() or ".." in Path(mesh).parts:
        raise ValueError(f"Unexpected mesh path: {mesh}")
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, sorted(meshes)))
fetch("LICENSE", "LICENSE")
source_record.write_text(json.dumps({"repository": "TheRobotStudio/SO-ARM100", "revision": revision}, indent=2))
print(f"SO-101 model ready: {urdf.resolve()} ({len(meshes)} meshes, revision {revision})")
PY

uv run --no-project --python .venv-gamepad/bin/python \
    python -m lerobot.rl.gym_manipulator --help >/dev/null
printf '%s\n' 'Setup complete. Launch instructions: examples/gamepad_to_so101/README.md'
