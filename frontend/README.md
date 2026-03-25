# Frontend

This is the React + Vite client for the knuckle biometric system.

## Features

- registration form with knuckle image upload
- UID-based verification flow
- 1:N search against the registered database
- admin database view and delete actions
- shared API base URL through `VITE_API_URL`

## Run

```bash
cd frontend
npm install
npm run dev
```

## Build

```bash
cd frontend
npm run build
```

## Environment

Use `.env.example` as the frontend environment template:

```bash
VITE_API_URL=http://localhost:5001
```

Point this value at the backend server started from `backend/server.py`.
