"""
Shan AI - Image Generation Agent
Groq (LLM brain) + NVIDIA NIM (image engine)
Run: python app.py  — then open http://localhost:5000
"""

import os, base64, json, time, uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response
import requests
from groq import Groq

app = Flask(__name__)

BASE_DIR    = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── API clients ────────────────────────────────────────────────────────────────
def get_groq_client():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=key)

def get_nvidia_key():
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise ValueError("NVIDIA_API_KEY not set")
    return key

# ── NIM model config ───────────────────────────────────────────────────────────
# build.nvidia.com cloud API — correct genai endpoint, one URL per model
NIM_MODELS = {
    "flux-schnell": {
        "url":   "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
        "label": "FLUX.1 Schnell",
        "desc":  "Fastest · great quality"
    },
    "flux-dev": {
        "url":   "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
        "label": "FLUX.1 Dev",
        "desc":  "Best quality · slower"
    },
    "sdxl": {
        "url":   "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl",
        "label": "Stable Diffusion XL",
        "desc":  "Classic · versatile"
    },
    "sdxl-turbo": {
        "url":   "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo",
        "label": "SDXL Turbo",
        "desc":  "Ultra-fast · lighter"
    }
}

STYLE_PRESETS = {
    "photorealistic": "photorealistic, ultra detailed, 8k, professional photography, sharp focus",
    "cinematic":      "cinematic, dramatic lighting, film grain, anamorphic lens, movie still",
    "anime":          "anime style, vibrant colors, clean lines, studio ghibli inspired, illustrated",
    "digital-art":    "digital art, concept art, artstation, highly detailed illustration",
    "oil-painting":   "oil painting, impressionist, textured brushstrokes, classical fine art",
    "minimalist":     "minimalist design, clean, simple, geometric, soft lighting, white space",
    "dark-fantasy":   "dark fantasy, moody, gothic, dramatic shadows, epic atmosphere",
    "none": ""
}

# ── Step 1: Groq expands the prompt ───────────────────────────────────────────
def expand_prompt_with_groq(user_input: str, style: str, aspect: str) -> dict:
    client     = get_groq_client()
    style_hint = STYLE_PRESETS.get(style, "")
    # Supported sizes for NVIDIA NIM images API
    aspect_map = {
        "1:1":  "1024x1024",
        "16:9": "1280x720",
        "9:16": "720x1280",
        "4:3":  "1024x768",
        "3:2":  "1152x768",
    }
    size = aspect_map.get(aspect, "1024x1024")
    w, h = size.split("x")

    system_prompt = f"""You are an expert AI image prompt engineer specializing in diffusion models.
Given the user's idea, craft a detailed, vivid image generation prompt.
{"Apply this style: " + style_hint if style_hint else ""}
Return ONLY valid JSON with these exact keys:
- prompt: rich positive prompt (80-150 words, descriptive, specific details, artistic)
- negative_prompt: comma-separated list of things to avoid (e.g. blurry, watermark, text)
- seed: a random integer between 1 and 99999
No markdown fences. No extra keys."""

    response = get_groq_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_input}
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=512
    )
    params = json.loads(response.choices[0].message.content)
    params["size"]  = size
    params["width"] = int(w)
    params["height"]= int(h)
    params.setdefault("steps", 25)
    params.setdefault("guidance_scale", 7.5)
    params.setdefault("seed", int(time.time()) % 99999)
    return params

# ── Step 2: NVIDIA NIM generates image ────────────────────────────────────────
def generate_with_nim(params: dict, model_key: str) -> bytes:
    model_cfg = NIM_MODELS.get(model_key, NIM_MODELS["flux-schnell"])
    url       = model_cfg["url"]

    headers = {
        "Authorization": f"Bearer {get_nvidia_key()}",
        "Accept":        "application/json",
        "Content-Type":  "application/json"
    }

    # FLUX.1 Schnell cloud API only accepts these 4 fields — no extras allowed
    payload = {
        "prompt": params["prompt"],
        "seed":   int(params.get("seed", 0)),
        "width":  params.get("width",  1024),
        "height": params.get("height", 1024),
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if not resp.ok:
        try:
            err_detail = resp.json()
        except Exception:
            err_detail = resp.text[:300]
        raise requests.HTTPError(
            f"NVIDIA NIM {resp.status_code}: {json.dumps(err_detail)[:400]}",
            response=resp
        )

    body = resp.json()
    # genai API response: {"artifacts": [{"base64": "...", "finishReason": "SUCCESS"}]}
    b64 = body["artifacts"][0]["base64"]
    return base64.b64decode(b64)


# ── HTML (embedded) ────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shan AI · Image Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0b;--surface:#111114;--card:#17171c;--border:#26262e;--border2:#32323c;--amber:#f0a500;--teal:#00c9a7;--text:#e8e6e0;--muted:#6e6c78;--muted2:#9a98a6;--serif:'Instrument Serif',Georgia,serif;--mono:'DM Mono',monospace;--sans:'Outfit',sans-serif;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden;}
header{position:sticky;top:0;z-index:100;background:rgba(10,10,11,0.9);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 32px;display:flex;align-items:center;justify-content:space-between;height:60px;}
.logo{display:flex;align-items:center;gap:10px;}
.logo-mark{width:32px;height:32px;background:linear-gradient(135deg,var(--amber),var(--teal));border-radius:8px;display:grid;place-items:center;font-family:var(--serif);font-size:18px;color:#000;font-weight:700;}
.logo-text{font-size:15px;font-weight:500;}
.logo-text span{color:var(--amber);}
.header-badge{font-family:var(--mono);font-size:11px;color:var(--muted);border:1px solid var(--border2);border-radius:20px;padding:3px 10px;}
.shell{display:grid;grid-template-columns:360px 1fr;min-height:calc(100vh - 60px);}
.sidebar{border-right:1px solid var(--border);padding:24px 20px;display:flex;flex-direction:column;gap:20px;background:var(--surface);overflow-y:auto;}
.section-label{font-family:var(--mono);font-size:10px;letter-spacing:.14em;color:var(--muted);text-transform:uppercase;margin-bottom:8px;}
textarea,select{width:100%;background:var(--card);border:1px solid var(--border2);border-radius:10px;color:var(--text);font-family:var(--sans);font-size:14px;transition:border-color .2s;}
textarea:focus,select:focus{outline:none;border-color:var(--amber);}
textarea{padding:12px;resize:vertical;min-height:100px;line-height:1.6;}
select{padding:10px 14px;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%236e6c78' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center;}
option{background:var(--card);}
.aspect-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;}
.aspect-pill{border:1px solid var(--border2);border-radius:8px;background:var(--card);color:var(--muted2);padding:8px 4px;text-align:center;font-family:var(--mono);font-size:11px;cursor:pointer;transition:all .15s;}
.aspect-pill:hover{border-color:var(--amber);color:var(--text);}
.aspect-pill.active{border-color:var(--amber);background:rgba(240,165,0,.12);color:var(--amber);font-weight:500;}
.style-grid{display:flex;flex-wrap:wrap;gap:6px;}
.style-chip{border:1px solid var(--border2);border-radius:20px;background:var(--card);color:var(--muted2);padding:5px 12px;font-size:12px;cursor:pointer;transition:all .15s;}
.style-chip:hover{border-color:var(--teal);color:var(--text);}
.style-chip.active{border-color:var(--teal);background:rgba(0,201,167,.1);color:var(--teal);}
.btn-generate{width:100%;padding:14px;background:linear-gradient(135deg,var(--amber) 0%,#e8900a 100%);color:#000;border:none;border-radius:12px;font-family:var(--sans);font-size:15px;font-weight:600;cursor:pointer;transition:transform .15s,opacity .15s;}
.btn-generate:hover:not(:disabled){transform:translateY(-1px);opacity:.92;}
.btn-generate:disabled{opacity:.5;cursor:not-allowed;transform:none;}
.btn-inner{display:flex;align-items:center;justify-content:center;gap:8px;}
.canvas{display:flex;flex-direction:column;background:var(--bg);}
.canvas-header{padding:20px 28px 0;display:flex;align-items:center;justify-content:space-between;}
.canvas-title{font-family:var(--serif);font-size:22px;font-style:italic;color:var(--muted2);}
.tab-row{display:flex;gap:0;border:1px solid var(--border);border-radius:10px;overflow:hidden;}
.tab{padding:7px 18px;font-size:13px;font-weight:500;background:transparent;color:var(--muted);border:none;cursor:pointer;transition:all .15s;font-family:var(--sans);}
.tab.active{background:var(--card);color:var(--text);}
.image-area{flex:1;display:flex;align-items:center;justify-content:center;padding:28px;min-height:480px;}
.placeholder{display:flex;flex-direction:column;align-items:center;gap:16px;color:var(--muted);}
.placeholder-icon{width:80px;height:80px;border-radius:20px;border:2px dashed var(--border2);display:grid;place-items:center;font-size:32px;}
.placeholder p{font-size:14px;text-align:center;max-width:240px;line-height:1.6;}
.result-card{display:none;flex-direction:column;gap:16px;width:100%;max-width:680px;align-items:center;}
.result-card.visible{display:flex;}
.result-img{max-width:100%;max-height:540px;border-radius:16px;border:1px solid var(--border);box-shadow:0 24px 80px rgba(0,0,0,.6);animation:fadeUp .4s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.result-meta{display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:center;}
.meta-pill{font-family:var(--mono);font-size:11px;border:1px solid var(--border2);border-radius:20px;padding:4px 12px;color:var(--muted2);}
.meta-pill.hl{border-color:var(--amber);color:var(--amber);}
.btn-download{padding:9px 20px;background:var(--card);border:1px solid var(--border2);border-radius:10px;color:var(--text);font-family:var(--sans);font-size:13px;cursor:pointer;transition:all .15s;font-weight:500;}
.btn-download:hover{border-color:var(--amber);color:var(--amber);}
.spinner-wrap{display:none;flex-direction:column;align-items:center;gap:20px;}
.spinner-wrap.visible{display:flex;}
.spinner{width:52px;height:52px;border-radius:50%;border:3px solid var(--border2);border-top-color:var(--amber);animation:spin .9s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.progress-steps{display:flex;flex-direction:column;gap:8px;margin-top:4px;}
.step{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--muted);font-family:var(--mono);transition:color .3s;}
.step.done{color:var(--teal);}
.step.active{color:var(--amber);}
.step-dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0;}
.prompt-reveal{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;margin:0 28px 16px;display:none;}
.prompt-reveal.visible{display:block;}
.prompt-reveal-label{font-family:var(--mono);font-size:10px;color:var(--amber);letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px;}
.prompt-reveal-text{font-size:13px;color:var(--muted2);line-height:1.65;}
.gallery-grid{display:none;padding:24px 28px;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;}
.gallery-grid.visible{display:grid;}
.gallery-thumb{aspect-ratio:1;border-radius:10px;overflow:hidden;border:1px solid var(--border);cursor:pointer;transition:transform .2s,border-color .2s;}
.gallery-thumb:hover{transform:scale(1.03);border-color:var(--amber);}
.gallery-thumb img{width:100%;height:100%;object-fit:cover;}
.gallery-empty{display:flex;align-items:center;justify-content:center;height:200px;color:var(--muted);font-size:13px;grid-column:1/-1;}
.toast{position:fixed;bottom:28px;left:50%;transform:translateX(-50%);background:#3d1212;border:1px solid #7a2020;border-radius:10px;padding:12px 20px;font-size:13px;color:#f87171;z-index:999;display:none;max-width:560px;text-align:center;word-break:break-word;}
.toast.visible{display:block;animation:fadeUp .3s ease;}
.hint{font-size:11px;color:var(--muted);line-height:1.6;}
.hint code{color:var(--amber);font-family:var(--mono);font-size:10px;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:transparent;}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-mark">S</div>
    <span class="logo-text">Shan <span>AI</span> &middot; Image Studio</span>
  </div>
  <div class="header-badge">Groq + NVIDIA NIM</div>
</header>
<div class="shell">
  <aside class="sidebar">
    <div>
      <div class="section-label">Your idea</div>
      <textarea id="promptInput" placeholder="Describe what you want to create&#10;e.g. ancient temple hidden in misty rainforest, golden hour light"></textarea>
    </div>
    <div>
      <div class="section-label">Image model</div>
      <select id="modelSelect">
        <option value="flux-schnell">FLUX.1 Schnell &mdash; Fastest &middot; great quality</option>
        <option value="flux-dev">FLUX.1 Dev &mdash; Best quality &middot; slower</option>
        <option value="sdxl">Stable Diffusion XL &mdash; Classic &middot; versatile</option>
        <option value="sdxl-turbo">SDXL Turbo &mdash; Ultra-fast &middot; lighter</option>
      </select>
    </div>
    <div>
      <div class="section-label">Style</div>
      <div class="style-grid" id="styleGrid">
        <div class="style-chip active" data-style="none">None</div>
        <div class="style-chip" data-style="photorealistic">Photo</div>
        <div class="style-chip" data-style="cinematic">Cinematic</div>
        <div class="style-chip" data-style="anime">Anime</div>
        <div class="style-chip" data-style="digital-art">Digital art</div>
        <div class="style-chip" data-style="oil-painting">Oil paint</div>
        <div class="style-chip" data-style="minimalist">Minimal</div>
        <div class="style-chip" data-style="dark-fantasy">Dark fantasy</div>
      </div>
    </div>
    <div>
      <div class="section-label">Aspect ratio</div>
      <div class="aspect-grid" id="aspectGrid">
        <div class="aspect-pill active" data-aspect="1:1">1:1</div>
        <div class="aspect-pill" data-aspect="16:9">16:9</div>
        <div class="aspect-pill" data-aspect="9:16">9:16</div>
        <div class="aspect-pill" data-aspect="4:3">4:3</div>
        <div class="aspect-pill" data-aspect="3:2">3:2</div>
      </div>
    </div>
    <button class="btn-generate" id="generateBtn" onclick="generate()">
      <div class="btn-inner">
        <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
        Generate Image
      </div>
    </button>
    <div class="hint">Tip: <strong>Ctrl+Enter</strong> to generate<br>Env vars: <code>GROQ_API_KEY</code> &amp; <code>NVIDIA_API_KEY</code></div>
  </aside>
  <main class="canvas">
    <div class="canvas-header">
      <span class="canvas-title">Canvas</span>
      <div class="tab-row">
        <button class="tab active" onclick="switchTab('generate')">Generate</button>
        <button class="tab" onclick="switchTab('gallery')">Gallery</button>
      </div>
    </div>
    <div id="generateView">
      <div class="image-area" id="imageArea">
        <div class="placeholder" id="placeholder">
          <div class="placeholder-icon">&#10022;</div>
          <p>Enter a prompt and hit Generate.<br>Your image will appear here.</p>
        </div>
        <div class="spinner-wrap" id="spinnerWrap">
          <div class="spinner"></div>
          <div class="progress-steps">
            <div class="step" id="step1"><span class="step-dot"></span>Sending prompt to Groq LLM&hellip;</div>
            <div class="step" id="step2"><span class="step-dot"></span>Crafting diffusion parameters&hellip;</div>
            <div class="step" id="step3"><span class="step-dot"></span>NVIDIA NIM generating image&hellip;</div>
            <div class="step" id="step4"><span class="step-dot"></span>Saving result&hellip;</div>
          </div>
        </div>
        <div class="result-card" id="resultCard">
          <img class="result-img" id="resultImg" src="" alt="Generated image">
          <div class="result-meta" id="resultMeta"></div>
          <button class="btn-download" onclick="downloadImage()">&#8595; Download PNG</button>
        </div>
      </div>
      <div class="prompt-reveal" id="promptReveal">
        <div class="prompt-reveal-label">Enhanced prompt (by Groq)</div>
        <div class="prompt-reveal-text" id="promptRevealText"></div>
      </div>
    </div>
    <div class="gallery-grid" id="galleryView"></div>
  </main>
</div>
<div class="toast" id="toast"></div>
<script>
let selStyle='none',selAspect='1:1',curFile=null;
document.querySelectorAll('.style-chip').forEach(c=>c.addEventListener('click',()=>{
  document.querySelectorAll('.style-chip').forEach(x=>x.classList.remove('active'));
  c.classList.add('active'); selStyle=c.dataset.style;
}));
document.querySelectorAll('.aspect-pill').forEach(p=>p.addEventListener('click',()=>{
  document.querySelectorAll('.aspect-pill').forEach(x=>x.classList.remove('active'));
  p.classList.add('active'); selAspect=p.dataset.aspect;
}));
function setStep(n){
  for(let i=1;i<=4;i++){const el=document.getElementById('step'+i);el.className='step';if(i<n)el.classList.add('done');if(i===n)el.classList.add('active');}
}
async function generate(){
  const prompt=document.getElementById('promptInput').value.trim();
  if(!prompt){showToast('Please enter a prompt first');return;}
  const btn=document.getElementById('generateBtn');
  btn.disabled=true;
  btn.querySelector('.btn-inner').innerHTML='<div style="width:16px;height:16px;border:2px solid #000;border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite"></div> Generating&hellip;';
  document.getElementById('placeholder').style.display='none';
  document.getElementById('resultCard').classList.remove('visible');
  document.getElementById('promptReveal').classList.remove('visible');
  document.getElementById('spinnerWrap').classList.add('visible');
  setStep(1);
  const t2=setTimeout(()=>setStep(2),1200);
  const t3=setTimeout(()=>setStep(3),2500);
  try{
    const res=await fetch('/api/generate',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt,model:document.getElementById('modelSelect').value,style:selStyle,aspect:selAspect})
    });
    const data=await res.json();
    clearTimeout(t2);clearTimeout(t3);
    if(!res.ok||!data.success){
      showToast(data.error||'Generation failed');
      document.getElementById('spinnerWrap').classList.remove('visible');
      document.getElementById('placeholder').style.display='flex';
    }else{
      setStep(4);
      await new Promise(r=>setTimeout(r,400));
      document.getElementById('spinnerWrap').classList.remove('visible');
      const img=document.getElementById('resultImg');
      img.src=data.image_url+'?t='+Date.now();
      curFile=data.filename;
      const p=data.params;
      document.getElementById('resultMeta').innerHTML=
        `<span class="meta-pill hl">${data.model}</span>
         <span class="meta-pill">${p.size}</span>
         <span class="meta-pill">${p.steps} steps</span>
         <span class="meta-pill">CFG ${p.guidance_scale}</span>
         <span class="meta-pill">seed ${p.seed}</span>`;
      document.getElementById('resultCard').classList.add('visible');
      document.getElementById('promptRevealText').textContent=p.prompt;
      document.getElementById('promptReveal').classList.add('visible');
    }
  }catch(e){
    clearTimeout(t2);clearTimeout(t3);
    showToast('Network error: '+e.message);
    document.getElementById('spinnerWrap').classList.remove('visible');
    document.getElementById('placeholder').style.display='flex';
  }
  btn.disabled=false;
  btn.querySelector('.btn-inner').innerHTML='<svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg> Generate Image';
}
function downloadImage(){
  if(!curFile)return;
  const a=document.createElement('a');a.href='/api/download/'+curFile;a.download=curFile;a.click();
}
async function loadGallery(){
  const el=document.getElementById('galleryView');el.innerHTML='';
  try{
    const data=await(await fetch('/api/gallery')).json();
    if(!data.length){el.innerHTML='<div class="gallery-empty">No images yet &mdash; generate your first!</div>';}
    else{data.forEach(img=>{
      const d=document.createElement('div');d.className='gallery-thumb';
      d.innerHTML=`<img src="${img.url}" alt="${img.name}" loading="lazy">`;
      d.onclick=()=>{switchTab('generate');document.getElementById('resultImg').src=img.url+'?t='+Date.now();curFile=img.name;document.getElementById('placeholder').style.display='none';document.getElementById('resultCard').classList.add('visible');document.getElementById('promptReveal').classList.remove('visible');};
      el.appendChild(d);
    });}
  }catch(e){el.innerHTML='<div class="gallery-empty">Could not load gallery</div>';}
}
function switchTab(tab){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',(i===0&&tab==='generate')||(i===1&&tab==='gallery')));
  document.getElementById('generateView').style.display=tab==='generate'?'block':'none';
  const gv=document.getElementById('galleryView');
  if(tab==='gallery'){gv.classList.add('visible');loadGallery();}else gv.classList.remove('visible');
}
function showToast(msg){
  const t=document.getElementById('toast');t.textContent=msg;t.classList.add('visible');
  setTimeout(()=>t.classList.remove('visible'),6000);
}
document.getElementById('promptInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&e.ctrlKey)generate();});
</script>
</body>
</html>"""

# ── Flask routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data       = request.json or {}
    user_input = data.get("prompt", "").strip()
    model_key  = data.get("model", "flux-schnell")
    style      = data.get("style", "none")
    aspect     = data.get("aspect", "1:1")
    if not user_input:
        return jsonify({"error": "Prompt is required"}), 400
    try:
        params      = expand_prompt_with_groq(user_input, style, aspect)
        image_bytes = generate_with_nim(params, model_key)
        filename    = f"img_{str(uuid.uuid4())[:8]}.png"
        (OUTPUTS_DIR / filename).write_bytes(image_bytes)
        return jsonify({"success": True,
                        "image_url": f"/outputs/{filename}",
                        "filename": filename,
                        "params": params,
                        "model": NIM_MODELS[model_key]["label"]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.HTTPError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_file(OUTPUTS_DIR / filename, mimetype="image/png")

@app.route("/api/gallery")
def api_gallery():
    imgs = sorted(OUTPUTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([{"url": f"/outputs/{p.name}", "name": p.name} for p in imgs[:24]])

@app.route("/api/download/<filename>")
def api_download(filename):
    return send_file(OUTPUTS_DIR / filename, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    print("\n  Shan AI - Image Generation Agent")
    print("  ----------------------------------")
    print("  Groq LLM      : llama-3.3-70b-versatile")
    print("  NVIDIA NIM API: integrate.api.nvidia.com")
    print("\n  Set env vars:")
    print("  set GROQ_API_KEY=gsk_...")
    print("  set NVIDIA_API_KEY=nvapi-...")
    print("\n  Open: http://localhost:5000\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
