# Visual Face Search API

Face matching API for modelsaler.com — Built with FastAPI, hosted on Railway.app

---

## Files in this Repository

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — handles face search requests |
| `requirements.txt` | Python libraries |
| `nixpacks.toml` | Build config (needed for dlib/cmake) |
| `railway.toml` | Railway deployment config |
| `.gitignore` | Ignores sensitive/unnecessary files |

> `face_database.json` is NOT uploaded to GitHub.
> You will upload it directly to Railway after building it.

---

## Step-by-Step Setup

### STEP 1 — Upload These Files to GitHub

1. Go to [github.com](https://github.com) → click **New Repository**
2. Name it: `face-search-api`
3. Set to **Private**
4. Click **Create Repository**
5. Click **uploading an existing file**
6. Drag and drop all 5 files:
   - `main.py`
   - `requirements.txt`
   - `nixpacks.toml`
   - `railway.toml`
   - `.gitignore`
7. Click **Commit changes**

---

### STEP 2 — Deploy on Railway

1. Go to [railway.app](https://railway.app) → Sign in with GitHub
2. Click **New Project**
3. Click **Deploy from GitHub Repo**
4. Select your `face-search-api` repository
5. Railway will start building automatically (takes 5–10 minutes)

---

### STEP 3 — Add Environment Variable on Railway

1. In Railway Dashboard → click your service
2. Click **Variables** tab
3. Click **Add Variable**
4. Add this:
   ```
   Name:  RELOAD_SECRET
   Value: mysecret123
   ```
   (You can use any password you want)

---

### STEP 4 — Get Your Railway URL

1. In Railway Dashboard → click your service
2. Click **Settings** tab
3. Under **Domains** → click **Generate Domain**
4. You will get a URL like:
   ```
   https://face-search-api-production-xxxx.railway.app
   ```
5. **Copy this URL** — you will need it for the WordPress plugin

---

### STEP 5 — Upload face_database.json to Railway

1. First run `build_face_database.py` on your computer
2. It will create `face_database.json`
3. In Railway Dashboard → your service → **Files** tab
4. Upload `face_database.json`
5. Railway will restart automatically

---

### STEP 6 — Test Your API

Open this URL in your browser:
```
https://your-app.railway.app/
```

You should see:
```json
{
  "status": "running",
  "models_loaded": 50,
  "message": "Visual Face Search API is live!"
}
```

If `models_loaded` is greater than 0 — everything is working!

---

## API Endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/` | Check if API is running |
| GET | `/health` | Health check |
| POST | `/search` | Upload image → find matching model |
| POST | `/reload-db?secret=XXX` | Reload database without restart |

---

## After Everything is Working

Tell your developer (or ask again) for the **WordPress PHP plugin** with iOS-style UI.
You will just need to paste your Railway URL into it — one line change.
