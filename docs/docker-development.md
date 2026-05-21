# SolarLab Docker Development

This setup runs the primary `perovskite-sim/` tree in two development
containers:

- `backend`: FastAPI + editable Python package on <http://127.0.0.1:8000>
- `frontend`: Vite + TypeScript UI on <http://127.0.0.1:5173>
- `notebook`: optional JupyterLab workspace on <http://127.0.0.1:8888>

The source tree is bind-mounted into the containers, so normal code edits on
the host are visible inside Docker. The backend uses `uvicorn --reload`; the
frontend uses Vite hot reload.

## First Run

From the SolarLab repository root:

```bash
docker compose up --build
```

Then open <http://127.0.0.1:5173>.

To open the numerical sweep notebook in the browser:

```bash
docker compose --profile notebook up --build notebook
```

Then open <http://127.0.0.1:8888/lab?token=solarlab> and start
`notebooks/08_device_parameter_sweep.ipynb`.

To run in the background:

```bash
docker compose up --build -d
docker compose logs -f
```

## Daily Commands

Start existing containers:

```bash
docker compose up
```

Stop containers:

```bash
docker compose down
```

Rebuild after dependency changes:

```bash
docker compose build --no-cache
docker compose up
```

Run backend tests inside the container:

```bash
docker compose run --rm backend pytest
```

Run one backend test:

```bash
docker compose run --rm backend pytest tests/unit/solver/test_mol.py
```

Run frontend build:

```bash
docker compose run --rm frontend npm run build
```

Run the device-parameter sweep CLI inside Docker:

```bash
docker compose run --rm backend python scripts/run_device_parameter_sweep.py \
  --preset pilot \
  --config configs/solarscale_nip_band_aligned.yaml \
  --out-dir outputs/device_parameter_sweep_pilot \
  --plots
```

Open a shell in a running container:

```bash
docker compose exec backend bash
docker compose exec frontend sh
```

## Updating To New Repo Changes

After pulling new code:

```bash
git pull
docker compose up --build
```

If only source code changed, `docker compose up` is usually enough because the
code is mounted into the containers. If `pyproject.toml`, `package.json`, or
`package-lock.json` changed, rebuild the images.

## Image-Style Run

For a clean image rebuild without bind-mounted development hot reload:

```bash
docker build -f perovskite-sim/Dockerfile.backend -t solarlab-backend:dev .
docker run --rm -p 8000:8000 solarlab-backend:dev
```

The Dockerfiles use Docker Official Images mirrored through AWS Public ECR
(`public.ecr.aws/docker/library/...`). This avoids intermittent Docker Hub
metadata and layer-download failures while keeping the same upstream images.

The full app is more convenient through `docker compose`, because SolarLab has
separate backend and frontend processes.
