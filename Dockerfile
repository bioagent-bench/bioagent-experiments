FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        build-essential \
        nodejs \
        npm \
        tree \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @openai/codex

RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin bin/micromamba \
    && ln -s /usr/local/bin/micromamba /usr/local/bin/mamba

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /workspace

COPY pyproject.toml README.md uv.lock /workspace/
COPY src /workspace/src
COPY otel.py /workspace/otel.py

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "src.agent", "--help"]
