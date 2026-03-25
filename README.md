# Knuckle Biometric Demo

This project now contains only the live demo app:

- `backend/` is the runtime API for register, verify, identify, and admin
- `frontend/` is the React UI

There is no training pipeline, no promoted checkpoint folder, and no bundled dataset dependency in the runtime path.

## Start

```bash
npm run setup
npm start
```

Frontend:

- open the Vite URL shown in the terminal

Backend:

- health: `http://127.0.0.1:5001/health`
- default admin code: `cbit`

## Backend Notes

- registration stores only user metadata, processed ROIs, embeddings, and one template under `backend/storage/`
- verification uses a built-in torchvision `EfficientNet-B0` feature extractor plus an ORB texture score
- the backend keeps the same API used by the existing frontend:
  - `POST /register`
  - `POST /verify`
  - `POST /identify`
  - `POST /admin/users`
  - `POST /admin/delete/:uid`
# Knuckle_MP_final
