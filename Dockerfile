FROM public.ecr.aws/docker/library/node:24-slim AS web-builder

WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build


FROM public.ecr.aws/docker/library/python:3.13-slim

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    NEWCORE_EQUATIONS=assembled \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY examples/research_1d/device.json ./examples/research_1d/device.json
COPY --from=web-builder /build/web/dist ./src/solarlab_server/_web_dist

RUN uv sync --frozen --extra server --no-dev && \
    mkdir -p /var/lib/solarlab/store /var/lib/solarlab/work

EXPOSE 8000

CMD ["/app/.venv/bin/python", "-m", "solarlab_server.cli", "--host", "0.0.0.0", "--port", "8000", "--store-dir", "/var/lib/solarlab/store", "--work-dir", "/var/lib/solarlab/work"]
