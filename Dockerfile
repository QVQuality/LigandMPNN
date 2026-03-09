FROM python:3.11-slim-bookworm

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# only copy requirements.txt for now to leverage Docker caching for dependency installation
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# setup uv venv (and use uv's venv as the "default" python environment)
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# install python deps
RUN uv pip install --no-cache-dir -r requirements.txt
RUN uv pip install "fastapi[standard]"

# copy the rest of the code
COPY . /app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
