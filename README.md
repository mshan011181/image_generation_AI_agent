# Shan AI · Image Studio

Standalone AI image generation app — **Groq** as the LLM brain, **NVIDIA NIM** as the image GPU engine.
No local GPU required. Runs entirely on your Windows PC via Git Bash.

---

## How It Works

```
Your idea (text)
      ↓
  [GROQ API]  — llama-3.3-70b-versatile
  Expands your prompt into a rich 100-word diffusion prompt
  + picks seed, width, height
      ↓
  [NVIDIA NIM API]  — FLUX.1 Schnell on H100 cloud GPUs
  Generates the image and returns it as base64 PNG
      ↓
  Saved to ./outputs/  →  displayed in browser
```

---

## Requirements

- Python 3.9+
- Git Bash on Windows (recommended) or CMD/PowerShell
- Groq API key (free) — https://console.groq.com
- NVIDIA NIM API key (free credits) — https://build.nvidia.com

---

## Setup — Git Bash (Recommended)

All commands below use Git Bash. On Windows, your `D:\` drive is `/d/` in Git Bash.

```bash
# Navigate to the project folder
cd /d/AI_____LABS_NEW/image_gen_ai_agent

# Install dependencies
pip install -r requirements.txt

# Set API keys (Git Bash uses export, not set)
export GROQ_API_KEY=gsk_your_key_here
export NVIDIA_API_KEY=nvapi_your_key_here

# Run the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Setup — Windows CMD Alternative

If not using Git Bash, use `set` instead of `export`:

```cmd
cd D:\AI_____LABS_NEW\image_gen_ai_agent
pip install -r requirements.txt
set GROQ_API_KEY=gsk_your_key_here
set NVIDIA_API_KEY=nvapi_your_key_here
python app.py
```

---

## Get Your API Keys

**Groq** (free, no credit card):
1. Go to https://console.groq.com
2. Sign up → API Keys → Create API Key
3. Copy the key starting with `gsk_`

**NVIDIA NIM** (free credits on signup):
1. Go to https://build.nvidia.com
2. Sign up → Profile icon → API Keys → Generate API Key
3. Copy the key starting with `nvapi-`
4. Free tier: 40 requests/minute — more than enough for image generation

---

## Project Structure

```
image_gen_ai_agent/
├── app.py            # Flask backend + full HTML UI embedded (no static folder needed)
├── requirements.txt  # flask, groq, requests
├── README.md         # This file
└── outputs/          # Generated PNG images saved here (auto-created on first run)
    └── img_*.png
```

> **Note:** The HTML UI is embedded directly inside `app.py` as a Python string and served
> via `Response(HTML, mimetype="text/html")`. There is no separate `static/` folder.
> This avoids Windows path issues where Flask's `send_from_directory` fails depending
> on which directory the terminal is launched from.

---

## Available Models

| Model | NIM Endpoint | Steps | Quality | Best For |
|---|---|---|---|---|
| FLUX.1 Schnell | black-forest-labs/flux.1-schnell | 1–4 | Excellent | Default — best speed/quality |
| FLUX.1 Dev | black-forest-labs/flux.1-dev | 20–50 | Best | Maximum quality |
| Stable Diffusion XL | stabilityai/stable-diffusion-xl | 20–30 | Good | Classic diffusion style |
| SDXL Turbo | stabilityai/sdxl-turbo | 1–4 | Decent | Rapid iteration |

**All models use the correct `ai.api.nvidia.com/v1/genai/` endpoint.**

---

## NVIDIA NIM API — Important Notes

The FLUX.1 Schnell cloud endpoint only accepts **4 fields** in the payload.
Sending any other fields (e.g. `negative_prompt`, `num_inference_steps`, `guidance_scale`)
returns HTTP 422 `extra_forbidden`.

```python
# Correct payload — only these 4 fields
payload = {
    "prompt": "...",   # expanded by Groq
    "seed":   12345,
    "width":  1024,
    "height": 1024,
}
# Response format: {"artifacts": [{"base64": "...", "finishReason": "SUCCESS"}]}
```

---

## Features

- Groq LLM auto-expands your simple idea into a rich diffusion prompt
- 8 style presets: None, Photo, Cinematic, Anime, Digital Art, Oil Paint, Minimal, Dark Fantasy
- 5 aspect ratios: 1:1 · 16:9 · 9:16 · 4:3 · 3:2
- 4-step animated progress indicator during generation
- Enhanced prompt revealed after each generation (so you can see what Groq wrote)
- Gallery tab — browse all previously generated images
- Download PNG button
- Ctrl+Enter shortcut to generate

---

## Output

All images are saved to `./outputs/` (relative to `app.py`'s location, not the terminal's
working directory). File names follow the pattern `img_XXXXXXXX.png` where `XXXXXXXX`
is a random 8-character UUID prefix.

---

## The "Development Server" Warning

When you run `python app.py`, Flask prints:

```
WARNING: This is a development server. Do not use it in a production deployment.
```

This is **normal and harmless** for local personal use. It's Flask reminding you not to
expose this server publicly without a proper WSGI server.

To silence it, install Waitress and change the last line of `app.py`:

```bash
pip install waitress
```

```python
# Replace app.run(...) with:
from waitress import serve
serve(app, host="0.0.0.0", port=5000)
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GROQ_API_KEY not set` | Env var missing in current shell | Re-run `export GROQ_API_KEY=...` in the same terminal |
| `NVIDIA_API_KEY not set` | Env var missing in current shell | Re-run `export NVIDIA_API_KEY=...` in the same terminal |
| Browser shows 404 on `/` | Wrong `app.py` version with static folder | Use the latest `app.py` — HTML is embedded, no static folder |
| NVIDIA NIM 404 | Wrong endpoint URL | Must use `ai.api.nvidia.com/v1/genai/` not `integrate.api.nvidia.com` |
| NVIDIA NIM 422 | Extra fields in payload | Only send `prompt`, `seed`, `width`, `height` — remove all other fields |
| Image generation slow | Normal for cloud GPU | FLUX.1 Schnell takes 8–15 seconds — expected behaviour |

---

## Cost

- **Groq:** $0.00 — llama-3.3-70b-versatile is free at this usage level
- **NVIDIA NIM:** Free credits on signup — sufficient for hundreds of FLUX.1 Schnell generations

---

*Built by Shanmugavelu · shanmaha.com · Shan AI Labs*
