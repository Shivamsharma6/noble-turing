# Deployment and Environment Configuration Guide

This guide provides details on how to deploy, configure, and verify the `noble-turing` external model service.

---

## Required Environment Variables

Configure the following environment variables before starting the service:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MACBOOK_API_KEY` | **Yes** | None | Secret key used for authenticating requests via the `X-API-Key` header. |
| `DATABASE_PATH` | No | `news_cache.db` | Path to the SQLite database. |
| `MODELS_DIR` | No | `models` | Directory where trained model directories and packages are stored. |
| `DATA_DIR` | No | `data` | Directory where uploaded dataset CSVs are stored. |
| `DEFAULT_ONNX_PARITY_TOLERANCE` | No | `0.0001` | Strict drift tolerance for ONNX model parity validation checks. |

---

## Startup Command

To launch the FastAPI service locally using the `uv` package manager:

```bash
export MACBOOK_API_KEY="my-secret-token"
export DATABASE_PATH="news_cache.db"

uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## Exposing on Local Network (LAN)

To expose the service to other machines on the local network (for example, Artha runtimes running on separate servers):

```bash
export MACBOOK_API_KEY="my-secret-token"
export DATABASE_PATH="news_cache.db"

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Example Artha Configuration

When integrating this service into your Artha setup, reference the base URL and the `X-API-Key` header:

```json
{
  "advisor": {
    "type": "external",
    "base_url": "http://<noble-turing-ip>:8000",
    "headers": {
      "X-API-Key": "my-secret-token"
    },
    "endpoints": {
      "health": "/health",
      "readiness": "/api/v1/readiness",
      "capabilities": "/api/v1/capabilities",
      "score_daily_mover": "/api/v1/score_daily_mover_candidates",
      "score_time_series": "/api/v1/score_time_series_candidates"
    }
  }
}
```

---

## Verification Commands

You can verify the status and capabilities of the running service using `curl`:

### 1. Health Endpoint (Unauthenticated)
```bash
curl http://127.0.0.1:8000/health
```
**Response**:
```json
{
  "status": "ok",
  "service": "noble-turing",
  "version": "0.1.0",
  "commit": "821256447814421b8c067831d102e3b1c676d1a2",
  "time": "2026-06-20T19:24:13.264356"
}
```

### 2. Readiness Endpoint (Authenticated)
```bash
curl -H "X-API-Key: my-secret-token" http://127.0.0.1:8000/api/v1/readiness
```

### 3. Capabilities Endpoint (Authenticated)
```bash
curl -H "X-API-Key: my-secret-token" http://127.0.0.1:8000/api/v1/capabilities
```
