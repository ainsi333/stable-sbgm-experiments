FROM python:3.12.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/opt/levy-exp3/src:/opt/levy-exp3

RUN groupadd --gid 10001 experiment \
    && useradd --uid 10001 --gid experiment --create-home experiment

WORKDIR /opt/levy-exp3
COPY containers/requirements-gpu-cu12.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.lock

COPY pyproject.toml README.md LICENSE THIRD_PARTY.md Makefile ./
COPY src ./src
COPY experiments ./experiments
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY slurm ./slurm
COPY reports ./reports
COPY containers ./containers
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && python -m pip check \
    && python -c "from pathlib import Path; import experiments.exp2, levy_experiments; root=Path('/opt/levy-exp3'); assert Path(experiments.exp2.__file__).is_relative_to(root); assert Path(levy_experiments.__file__).is_relative_to(root)" \
    && python -m pytest -q -m "not gpu"

USER experiment
ENTRYPOINT ["levy-exp3"]
CMD ["--help"]
