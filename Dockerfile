FROM mambaorg/micromamba

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml /app/

RUN micromamba install -y -n base -c conda-forge python=3.13 pip \
    && micromamba clean --all --yes \
    && pip install --no-cache-dir \
        "smolagents[docker,openai]>=1.21.2" \
        "docker>=7.1.0" \
        "openinference-instrumentation-smolagents>=0.1.16" \
        "arize-phoenix>=11.30.0"
