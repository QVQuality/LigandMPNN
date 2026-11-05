# The base and uv are pinned, so a rebuild changes only what constraints.txt changes.
FROM python:3.11.16-slim-bookworm

COPY --from=docker.io/astral/uv:0.12.21 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# only copy the dependency files for now to leverage Docker caching for dependency installation
WORKDIR /app
COPY requirements.txt constraints.txt /app/

# setup uv venv (and use uv's venv as the "default" python environment)
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# constraints.txt pins the whole install. --find-links adds PyTorch's CUDA 12.8 torch wheel only;
# everything else, the nvidia runtime wheels included, comes from PyPI.
RUN uv pip install --no-cache-dir -c constraints.txt -r requirements.txt \
    --find-links https://download.pytorch.org/whl/cu128/torch/

# copy the rest of the code
COPY . /app
