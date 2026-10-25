FROM python:3.10-slim-bookworm

# OSMesa renders inside the laptop's Linux container without an NVIDIA GPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libegl1 libglib2.0-0 libglfw3 ffmpeg g++ cmake make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY requirements-client.txt .
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2 \
    && uv pip install --system -r requirements-client.txt
COPY remote_protocol.py libero_client.py ./
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "libero_client.py"]
