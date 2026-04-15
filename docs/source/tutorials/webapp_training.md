# Web App Training Dashboard

Backend:

```bash
uvicorn webapp.backend.app:app --reload
```

Frontend:

```bash
cd webapp/frontend
npm install
npm run dev
```

Features:

- Upload dry/wet WAV files
- Launch training job with temporal parameters
- Poll logs and checkpoints
- Resume from checkpoint path
- Download exported `.nam` model
