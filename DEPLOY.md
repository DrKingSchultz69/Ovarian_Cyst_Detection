# Going live

Backend on Hugging Face Spaces, frontend on Vercel. Both free, both stay up
without your laptop or a Kaggle session.

Total: ~35 minutes, most of it waiting on builds.

---

## Prerequisite — get the weights out of Kaggle

The one file nothing else can substitute.

1. Kaggle notebook → **Output** tab → download `ovascan_burnin1.pt`
2. Put it at `backend/weights/ovascan_burnin1.pt`

It is ~6 MB, so it commits straight into git. No Git LFS, no object storage.

```bash
ls -lh backend/weights/
```

---

## Step 1 — Backend on Hugging Face Spaces (~15 min)

1. [huggingface.co/new-space](https://huggingface.co/new-space) → name `ovascan-api`
   → **SDK: Docker** → **Blank** template → Public → Create
2. Clone it and copy the backend in:

```bash
git clone https://huggingface.co/spaces/<your-username>/ovascan-api
cd ovascan-api
cp ../ovascan/backend/{Dockerfile,requirements.txt,inference.py,main.py} .
mkdir -p weights && cp ../ovascan/backend/weights/ovascan_burnin1.pt weights/
```

3. Spaces needs YAML front-matter in its README to pick the right SDK and port:

```bash
printf -- '---\ntitle: OvaScan API\nemoji: 🩺\ncolorFrom: blue\ncolorTo: gray\nsdk: docker\napp_port: 7860\npinned: false\n---\n\nOvaScan segmentation API. Research prototype, not a medical device.\n' > README.md
```

4. Push:

```bash
git add -A && git commit -m "OvaScan API" && git push
```

Watch the build in the Space's **Logs** tab. First build is ~8 minutes (torch is
large). When it goes green:

```bash
curl https://<your-username>-ovascan-api.hf.space/health
```

That URL is your `NEXT_PUBLIC_API_URL`.

---

## Step 2 — Frontend on Vercel (~10 min)

1. Push `ovascan/frontend` to a GitHub repo
2. [vercel.com/new](https://vercel.com/new) → import it → **Root Directory: `frontend`**
3. Environment Variables → add `NEXT_PUBLIC_API_URL` = your Space URL, no trailing slash
4. Deploy

`NEXT_PUBLIC_*` variables are inlined at build time, so **changing it later needs a
redeploy**, not just a save.

---

## Step 3 — Lock down CORS (~5 min)

`OVASCAN_ALLOWED_ORIGINS` defaults to `*`, which is fine for a prototype and wrong
for a public URL. In the Space: **Settings → Variables and secrets → New variable**:

```
OVASCAN_ALLOWED_ORIGINS = https://<your-project>.vercel.app
```

The Space restarts itself. Test the live site afterwards — if uploads start
failing with a network error, the origin string does not match exactly (scheme,
subdomain and no trailing slash all matter).

---

## What to expect once live

| | Value |
|---|---|
| Cold start | ~30 s after 48 h idle (free tier sleeps) |
| Warm inference | ~150 ms CPU, vs 2.4 ms on the Kaggle T4 |
| Concurrency | 1 worker, ~500 MB resident; requests queue rather than fail |

Response sizes, measured on the 640×480 phantom. The analysis endpoints return
**uncompressed PNG panels**, which is deliberate — the frontend computes intensity
histograms from them client-side, and a lossy re-encode would put JPEG artifacts into
the very histograms that are meant to show what a filter did.

| Endpoint | Size | Time |
|---|---|---|
| `GET /health`, `/api` | < 1 KB | ~6 ms |
| `GET /image/phantom` | 0.6 MB | 0.2 s |
| `POST /predict` | ~0.5 MB, ~1.9 MB with debug panels | 0.15 s |
| `POST /image/register` | 3.1 MB | 1.1 s |
| `GET /image/register/demo` | 4.3 MB | 3.7 s |
| `POST /image/enhance` | 4.7 MB | 1.5 s |
| `POST /image/compress` | 1.8 MB | 5.2 s |
| `GET /ehr/pipeline` | 46 KB | 3.5 s first call, then cached |

Two consequences for a free-tier host. `POST /image/enhance` moves ~5 MB per request,
so it is not something to put behind an auto-refreshing control. And
`POST /image/compress` spends five seconds encoding the same frame at ten quality
levels plus a binary search per codec — fine on demand, wrong in a loop.

The EHR pipeline is cached per `(n_patients, seed)`, so only the first call pays the
3.5 s; changing the seed in the UI pays it again.

Hit `/health` once before any demo to wake the Space.

---

## Keeping the Kaggle path

The ngrok tunnel still works and is faster (GPU). Keep it as the demo path and
the Space as the always-on link — switching is one env var and a redeploy.
