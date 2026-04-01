# WeatherMind — Vercel Deployment Guide

## Project structure Vercel expects

```
weathermind-vercel/
├── api/
│   └── index.py          ← FastAPI backend (Vercel Serverless Function)
├── public/
│   └── index.html        ← Frontend (served by Vercel CDN)
├── requirements.txt      ← Python dependencies
├── vercel.json           ← Routing config
├── .gitignore
└── .env                  ← LOCAL ONLY — never commit this
```

---

## Step 1 — Install Vercel CLI

```bash
npm install -g vercel
```

---

## Step 2 — Push to GitHub

```bash
cd weathermind-vercel
git init
git add .
git commit -m "Initial WeatherMind deploy"
```

Create a new repo on github.com, then:
```bash
git remote add origin https://github.com/YOUR_USERNAME/weathermind.git
git push -u origin main
```

---

## Step 3 — Deploy via Vercel CLI

```bash
vercel
```

Follow the prompts:
- Set up and deploy? → **Y**
- Which scope? → your account
- Link to existing project? → **N**
- Project name → `weathermind` (or anything)
- In which directory is your code? → `./`
- Want to override settings? → **N**

Vercel will give you a preview URL like:
`https://weathermind-abc123.vercel.app`

---

## Step 4 — Add your AccuWeather API key as an environment variable

**Option A — Vercel CLI:**
```bash
vercel env add ACCUWEATHER_API_KEY
```
When prompted, paste your key. Set it for: Production, Preview, Development.

**Option B — Vercel Dashboard:**
1. Go to vercel.com → your project → Settings → Environment Variables
2. Add:
   - Key:   `ACCUWEATHER_API_KEY`
   - Value: `your_actual_key_here`
   - Environments: ✅ Production  ✅ Preview  ✅ Development

---

## Step 5 — Redeploy with the env variable active

```bash
vercel --prod
```

Your live URL will be:
`https://weathermind.vercel.app` (or your chosen name)

---

## How it works on Vercel (vs localhost)

| | Localhost | Vercel |
|--|-----------|--------|
| Backend | `uvicorn` long-running process | Serverless function (spins up per request) |
| ML model | Trained once, cached in memory | Trained fresh per `/api/ml/yearly` call |
| Frontend | Open HTML file directly | Served from Vercel CDN via `/public/` |
| API URL | `http://localhost:8000/api/...` | `https://your-app.vercel.app/api/...` |
| Env vars | `.env` file | Vercel dashboard / CLI |

**Important:** Because Vercel is serverless, the ML model cannot be cached between
requests. Each call to `/api/ml/yearly` fetches 5 years of Open-Meteo data and
trains fresh. This takes ~15-25 seconds. The frontend handles this gracefully —
weather data loads instantly while ML results arrive asynchronously.

---

## Vercel function limits (free Hobby plan)

| Limit | Value |
|-------|-------|
| Execution time | 60 seconds |
| Memory | 1024 MB |
| Bundle size | 250 MB |
| Bandwidth | 100 GB/month |
| Serverless invocations | 100,000/month |

The ML training (~15-25s) stays within the 60s limit.

---

## Testing after deploy

```bash
# Health check
curl https://your-app.vercel.app/api/health

# City search
curl "https://your-app.vercel.app/api/search?q=islamabad"

# Full city data
curl "https://your-app.vercel.app/api/city/full?key=187745&lat=33.72&lon=73.06"

# ML yearly prediction (slow — trains model)
curl "https://your-app.vercel.app/api/ml/yearly?lat=33.72&lon=73.06"
```

---

## Updating the app

Any `git push` to your main branch auto-deploys via Vercel CI/CD:
```bash
git add .
git commit -m "Update"
git push
```

Vercel builds and deploys automatically in ~30 seconds.
