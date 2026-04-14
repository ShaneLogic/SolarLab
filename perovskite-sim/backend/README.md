# Backend — FastAPI HTTP API

Thin FastAPI wrapper around the `perovskite_sim` library. The frontend at
`perovskite-sim/frontend/` talks to this service over HTTP + Server-Sent
Events. For the full project overview, physics, and UI guide see the
[root README](../../README.md).

## Running

From the **SolarLab root**:

```bash
uvicorn backend.main:app \
    --host 127.0.0.1 --port 8000 \
    --app-dir perovskite-sim --reload
```

> ⚠️ Do **not** run `uvicorn main:app` from inside `backend/` — module
> imports resolve against `--app-dir`, and the shorter form breaks them.
> Also: `--reload` occasionally misses edits under `backend/`; if a new
> endpoint returns 404, kill and restart rather than trusting the watcher.

## Endpoint map

### Configuration

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/configs` | List shipped + user presets (auto-scans `configs/`) |
| `GET` | `/api/configs/{name}` | Load one device preset (numerics coerced via `_coerce_numbers`) |
| `POST` | `/api/configs/user` | Save a user-edited stack to `configs/user/` |
| `GET` | `/api/layer-templates` | Layer template library for the Layer Builder UI |
| `GET` | `/api/optical-materials` | Available `optical_material` keys for TMM |

### Experiments — streaming (preferred)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Submit a job `{kind, config_path\|device, params}` → `{job_id}` |
| `GET` | `/api/jobs/{id}/events` | SSE stream: `progress` · `result` · `error` · `done` |

The SSE stream uses **named events**, not the default message channel.
Progress frames are JSON `{stage, current, total, eta_s, message}`. The
handler emits `: keepalive` comments between progress frames and drains
the job queue via `run_in_executor` with a 0.5 s timeout so it never
blocks the event loop.

### Experiments — legacy sync (back-compat)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jv` | Blocking J–V sweep |
| `POST` | `/api/impedance` | Blocking impedance sweep |
| `POST` | `/api/degradation` | Blocking degradation run |

These run synchronously and return the full result in one response. The
frontend no longer uses them; they exist for notebook / CLI consumers.

## Internal plumbing

- **`jobs.py`** — thread-per-job registry. `JobRegistry.submit()` runs the
  experiment on a background thread and hands each call a
  `ProgressReporter` it can emit frames through.
- **`progress.py`** — pub/sub primitive backing the SSE stream. Any new
  experiment that wants progress should accept
  `progress: Callable[[str, int, int, str], None] | None = None`; the
  `_run(reporter)` closure in `main.py` wraps `reporter.report` and
  passes it as that kwarg.
- **`_coerce_numbers`** — recursive pass applied in `get_config` that
  turns any numeric-looking string (including bare `1e-9` which PyYAML
  1.1 leaves as a string) into a float. Do not bypass it in new loaders.

## Configuration auto-discovery

`GET /api/configs` scans `perovskite-sim/configs/` on every request, so
dropping a new YAML file in that folder makes it immediately visible to
the frontend dropdown with no code change. User presets saved via
`POST /api/configs/user` land in `configs/user/` and are returned in a
separate `namespace: "user"` group.

## Experimental — Celery

`celery_app.py` and `requirements-celery.txt` are kept as a reference
for a future distributed-worker backend but are **not on the active
code path**. The frontend, tests, and `/api/jobs` endpoint all use the
in-process thread registry in `jobs.py`. Do not assume a Redis broker
is available.
