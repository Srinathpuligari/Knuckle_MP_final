# Backend

The backend is now a small self-contained runtime service.

## Files

- `server.py`: HTTP API used by the frontend
- `main.py`: simple entrypoint alias for the server
- `config.py`: runtime settings and thresholds
- `image_pipeline.py`: ROI extraction, alignment, and CLAHE preprocessing
- `feature_extractor.py`: built-in EfficientNet-B0 and lightweight texture descriptors
- `engine.py`: enrollment and comparison logic
- `storage.py`: SQLite-backed registration storage

## Run

```bash
./.venv/bin/python backend/server.py
```

Optional flags:

- `--host 0.0.0.0`
- `--port 5001`
- `--admin-code cbit`

## Storage

- runtime data lives in `backend/storage/`
- the database is created automatically on first start
- deleting `backend/storage/registry.sqlite3` resets enrollments
