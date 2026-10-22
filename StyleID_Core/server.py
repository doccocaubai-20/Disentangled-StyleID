# Paths work independently of the terminal's current directory.
from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_HERE / '.env', override=False)
_DEFAULT_RESOURCES = _HERE if (_HERE/'SKSF-A_data').is_dir() else _HERE.parent
RESOURCE_ROOT = _Path(_os.environ.get('STYLEID_RESOURCES', str(_DEFAULT_RESOURCES))).resolve()
_CKPT_OPTIONS = [_HERE/'checkpoints'/'soup', _HERE, RESOURCE_ROOT]
_DEFAULT_CKPT = next((p for p in _CKPT_OPTIONS if (p/'disentangled_lora_clip_SOUP_S6_A100.pt').is_file()), RESOURCE_ROOT)
CHECKPOINT_ROOT = _Path(_os.environ.get('STYLEID_CHECKPOINT_DIR', str(_DEFAULT_CKPT))).resolve()
_os.chdir(RESOURCE_ROOT)

# -*- coding: utf-8 -*-
"""
==============================================================================
HỆ THỐNG ĐÁNH GIÁ THỰC NGHIỆM: STYLEID vs DISENTANGLED-STYLEID (MODEL SOUP)
Báo cáo đề tài: Nhận diện khuôn mặt dưới tác động phong cách hóa
Khởi chạy: python web_demo.py
Truy cập: http://localhost:8000
==============================================================================
"""
import os
import sys
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import webbrowser
import threading
from pathlib import Path
from demo_leace import DemoLeace

DEMO_ROOT = Path(globals().get('RESOURCE_ROOT', Path(__file__).resolve().parent))
os.chdir(DEMO_ROOT)

# Cấu hình encoding UTF-8
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ------------------------------------------------------------------------------
# 1. KIẾN TRÚC MÔ HÌNH
# ------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=1.0):
        super().__init__()
        self.linear, self.scaling = linear_layer, alpha / r
        self.lora_down = nn.Linear(linear_layer.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, linear_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
    def forward(self, x):
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling

def apply_lora_to_clip(model, r=8):
    for layer in model.vision_model.encoder.layers:
        attn = layer.self_attn
        attn.q_proj = LoRALinear(attn.q_proj, r=r).to(device)
        attn.k_proj = LoRALinear(attn.k_proj, r=r).to(device)
        attn.v_proj = LoRALinear(attn.v_proj, r=r).to(device)
        attn.out_proj = LoRALinear(attn.out_proj, r=r).to(device)

class DisentangledHead(nn.Module):
    def __init__(self, clip_dim=768, id_dim=512, style_dim=256):
        super().__init__()
        self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
    def forward(self, feat):
        return F.normalize(self.id_proj(feat), dim=-1)

def load_image_clean(image_or_path):
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path)
    else:
        img = image_or_path
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        canvas = Image.new('RGB', img.size, (255, 255, 255))
        canvas.paste(img, mask=img.split()[3])
        return canvas
    return img.convert('RGB')

# ------------------------------------------------------------------------------
# 2. NẠP MÔ HÌNH VÀ BỘ TIỀN XỬ LÝ VÀO BỘ NHỚ
# ------------------------------------------------------------------------------
print("================================================================================")
print("KHOI DONG HE THONG SO SANH THUC NGHIEM STYLEID vs DISENTANGLED-STYLEID")
print(f"Thiet bi tinh toan: {device.upper()} ({device_name})")
print("================================================================================")

BASE_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
STYLEID_REVISION = '1967c354f339a636e5b3e16ecab3d0075aa27ab1'
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', revision=BASE_REVISION)

# A. Mô hình đối chứng: StyleID Chính chủ
print("--> Nạp mô hình đối chứng: StyleID (kwanY/styleid)...")
official_model = CLIPModel.from_pretrained('kwanY/styleid', revision=STYLEID_REVISION, attn_implementation='eager').to(device)
official_model.eval()

# B. Mô hình đề tài: Disentangled-StyleID (Model Soup)
print("--> Nạp mô hình đề tài: Disentangled-StyleID (Model Soup)...")
soup_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', revision=BASE_REVISION, attn_implementation='eager').to(device)
for p in soup_model.parameters(): p.requires_grad = False
apply_lora_to_clip(soup_model, r=8)
soup_head = DisentangledHead().to(device)

soup_lora_path = str(CHECKPOINT_ROOT/'disentangled_lora_clip_SOUP_S6_A100.pt')
soup_head_path = str(CHECKPOINT_ROOT/'disentangled_head_SOUP_S6_A100.pt')
if not os.path.isfile(soup_lora_path) or not os.path.isfile(soup_head_path):
    raise FileNotFoundError('Missing exact Soup files. Set STYLEID_CHECKPOINT_DIR; no shard fallback.')
model_display_name = 'Disentangled-StyleID (Model Soup)'

soup_model.load_state_dict(torch.load(soup_lora_path, map_location=device, weights_only=True), strict=True)
head_state = torch.load(soup_head_path, map_location=device, weights_only=True)
soup_head.load_state_dict({k:v for k,v in head_state.items() if k.startswith('id_proj.')}, strict=True)
soup_model.eval()
soup_head.eval()

# Làm ấm mô hình
dummy_img = Image.new('RGB', (224, 224), (128, 128, 128))
dummy_pv = processor(images=[dummy_img], return_tensors='pt')['pixel_values'].to(device)
with torch.no_grad():
    _ = official_model.get_image_features(pixel_values=dummy_pv)
    _f = soup_model.visual_projection(soup_model.vision_model(pixel_values=dummy_pv).pooler_output)
    _ = soup_head(_f)
print("--> Khoi dong hoan tat, he thong san sang.")

# ------------------------------------------------------------------------------
# 3. FASTAPI BACKEND
# ------------------------------------------------------------------------------
app = FastAPI(title="Face Verification Benchmark")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img_path):
    if not os.path.exists(img_path): return None
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(img_path)[1].lower().replace('.', '')
        if ext == 'jpg': ext = 'jpeg'
        return f"data:image/{ext};base64,{encoded}"

PRESETS = {
    "1": {
        "title": "Cặp 1: Ảnh thật vs Tranh vẽ (Cùng danh tính)",
        "desc": "Cùng danh tính: Ảnh chụp thực tế đối chiếu với tranh nghệ thuật.",
        "ground_truth": "Cùng danh tính",
        "img1": "data/1.jpg",
        "img2": "data/2.jpg"
    },
    "2": {
        "title": "Cặp 2: SKSF-A Ký họa chì (Cùng danh tính)",
        "desc": "Cùng danh tính: Ảnh chân dung thực tế đối chiếu với tranh ký họa Style 4.",
        "ground_truth": "Cùng danh tính",
        "img1": "SKSF-A_data/Photo/1.png",
        "img2": "SKSF-A_data/style4/1.png"
    },
    "3": {
        "title": "Cặp 3: SKSF-A Khác người (Khác danh tính)",
        "desc": "Khác danh tính: Hai người khác nhau cùng vẽ theo phong cách Style 4.",
        "ground_truth": "Khác danh tính",
        "img1": "SKSF-A_data/Photo/1.png",
        "img2": "SKSF-A_data/style4/10.png"
    }
}

class CompareRequest(BaseModel):
    img1_base64: str
    img2_base64: str
    threshold: float = 0.40
    use_leace: bool = False

@app.get("/api/preset/{preset_id}")
async def get_preset(preset_id: str):
    if preset_id not in PRESETS:
        return JSONResponse({"error": "Không tìm thấy preset"}, status_code=404)
    item = PRESETS[preset_id]
    return {
        "id": preset_id,
        "title": item["title"],
        "desc": item["desc"],
        "ground_truth": item["ground_truth"],
        "img1": image_to_base64(item["img1"]),
        "img2": image_to_base64(item["img2"])
    }

@app.post("/api/compare")
async def compare_images(req: CompareRequest):
    try:
        if not req.img1_base64 or not req.img2_base64:
            return JSONResponse({"error": "Vui lòng chọn đủ 2 ảnh"}, status_code=400)

        b64_1 = req.img1_base64.split(",")[-1]
        im1 = load_image_clean(Image.open(io.BytesIO(base64.b64decode(b64_1))))

        b64_2 = req.img2_base64.split(",")[-1]
        im2 = load_image_clean(Image.open(io.BytesIO(base64.b64decode(b64_2))))

        pv1 = processor(images=im1, return_tensors='pt')['pixel_values'].to(device)
        pv2 = processor(images=im2, return_tensors='pt')['pixel_values'].to(device)

        with torch.no_grad():
            # StyleID Official
            f1_off = official_model.get_image_features(pixel_values=pv1)
            f2_off = official_model.get_image_features(pixel_values=pv2)
            e1_off = F.normalize(f1_off, dim=-1)
            e2_off = F.normalize(f2_off, dim=-1)
            sim_off = float((e1_off * e2_off).sum(dim=-1).item())

            # Disentangled-StyleID
            f1_my = soup_model.visual_projection(soup_model.vision_model(pixel_values=pv1).pooler_output)
            f2_my = soup_model.visual_projection(soup_model.vision_model(pixel_values=pv2).pooler_output)
            zid1 = soup_head(f1_my)
            zid2 = soup_head(f2_my)
            sim_my = float((zid1 * zid2).sum(dim=-1).item())

        filtered = leace_filter.apply(torch.cat((zid1, zid2)).detach().cpu().numpy())
        sim_leace = float(np.dot(filtered[0], filtered[1]))
        sim_selected = sim_leace if req.use_leace else sim_my

        def result(score):
            return {'similarity':score, 'is_same':bool(score >= req.threshold),
                    'label':'Cùng danh tính' if score >= req.threshold else 'Khác danh tính', 'dim':'512 chiều'}

        threshold = req.threshold
        delta = sim_selected - sim_off

        return {
            "threshold": threshold,
            "device": f"{device.upper()} ({device_name})",
            "model_name": "Soup + LEACE" if req.use_leace else "Soup gốc",
            "use_leace": req.use_leace,
            "soup_raw": result(sim_my),
            "soup_leace": result(sim_leace),
            "styleid": {
                "similarity": sim_off,
                "is_same": bool(sim_off >= threshold),
                "label": "Cùng danh tính" if sim_off >= threshold else "Khác danh tính",
                "dim": "768 chiều"
            },
            "soup": result(sim_selected),
            "comparison": {
                "delta": round(delta, 4)
            }
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ------------------------------------------------------------------------------
# 4. GIAO DIỆN WEB TRUNG TÍNH, CHUẨN MỰC HỌC THUẬT
# ------------------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disentangled-StyleID: Face Verification Benchmark</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
        .panel { background-color: #1e293b; border: 1px solid #334155; }
        .dropzone-active { border-color: #38bdf8 !important; background-color: #0c4a6e !important; }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between">

    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/90 py-3.5 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-wrap items-center justify-between gap-4">
            <div>
                <h1 class="text-base font-semibold text-slate-100 tracking-tight">
                    Disentangled-StyleID: Thực nghiệm Nhận diện Khuôn mặt Đa phong cách
                </h1>
                <p class="text-xs text-slate-400">Đối chiếu trực tiếp giữa mô hình StyleID (SIGGRAPH 2026) và Disentangled-StyleID</p>
            </div>
            <div class="flex items-center gap-2 text-xs">
                <span class="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300">
                    Mô hình: <span class="font-medium text-slate-100">Disentangled-StyleID (Soup)</span>
                </span>
                <span class="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300">
                    Thiết bị: <span id="deviceBadge" class="font-medium text-slate-100">CPU</span>
                </span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-1">
        
        <!-- Preset Selector -->
        <div class="panel rounded-lg p-4 mb-6">
            <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                <div class="text-xs font-medium text-slate-300">
                    Tập dữ liệu mẫu thử nghiệm:
                </div>
                <div class="flex flex-wrap gap-2">
                    <button onclick="loadPreset('1')" class="px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition">
                        Cặp 1: Ảnh thật vs Tranh vẽ (Cùng danh tính)
                    </button>
                    <button onclick="loadPreset('2')" class="px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition">
                        Cặp 2: SKSF-A Ký họa chì (Cùng danh tính)
                    </button>
                    <button onclick="loadPreset('3')" class="px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition">
                        Cặp 3: SKSF-A Khác người (Khác danh tính)
                    </button>
                </div>
            </div>
            <div id="presetDescBox" class="mt-2.5 pt-2.5 border-t border-slate-700/60 flex items-center justify-between text-xs text-slate-400">
                <span id="presetDesc">Mô tả: Chưa chọn mẫu thử</span>
                <span id="presetGroundTruth" class="font-medium text-slate-300"></span>
            </div>
        </div>

        <!-- 2 Image Dropzones -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <!-- Image 1 -->
            <div class="panel rounded-lg p-4 flex flex-col justify-between">
                <div class="flex items-center justify-between mb-2.5">
                    <span class="text-xs font-semibold text-slate-300">Ảnh 1: Khuôn mặt tham chiếu</span>
                    <button onclick="clearImage(1)" id="btnClear1" class="text-xs text-slate-500 hover:text-slate-300 hidden">Xóa</button>
                </div>
                <div id="dropzone1" onclick="document.getElementById('fileInput1').click()" class="border border-dashed border-slate-700 hover:border-slate-500 rounded p-4 flex flex-col items-center justify-center min-h-[240px] cursor-pointer transition bg-slate-900/50">
                    <input type="file" id="fileInput1" accept="image/*" class="hidden" onchange="handleFileSelect(1, this.files)">
                    <div id="placeholder1" class="text-center">
                        <p class="text-xs text-slate-400">Chọn hoặc kéo thả ảnh vào đây</p>
                    </div>
                    <img id="preview1" class="hidden max-h-[220px] max-w-full object-contain rounded">
                </div>
            </div>

            <!-- Image 2 -->
            <div class="panel rounded-lg p-4 flex flex-col justify-between">
                <div class="flex items-center justify-between mb-2.5">
                    <span class="text-xs font-semibold text-slate-300">Ảnh 2: Khuôn mặt đối chiếu (Phong cách hóa)</span>
                    <button onclick="clearImage(2)" id="btnClear2" class="text-xs text-slate-500 hover:text-slate-300 hidden">Xóa</button>
                </div>
                <div id="dropzone2" onclick="document.getElementById('fileInput2').click()" class="border border-dashed border-slate-700 hover:border-slate-500 rounded p-4 flex flex-col items-center justify-center min-h-[240px] cursor-pointer transition bg-slate-900/50">
                    <input type="file" id="fileInput2" accept="image/*" class="hidden" onchange="handleFileSelect(2, this.files)">
                    <div id="placeholder2" class="text-center">
                        <p class="text-xs text-slate-400">Chọn hoặc kéo thả ảnh vào đây</p>
                    </div>
                    <img id="preview2" class="hidden max-h-[220px] max-w-full object-contain rounded">
                </div>
            </div>
        </div>

        <!-- Controls -->
        <div class="panel rounded-lg p-4 mb-6">
            <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                <div class="w-full md:w-1/2">
                    <div class="flex justify-between items-center mb-1.5">
                        <label class="text-xs font-medium text-slate-300">Ngưỡng phân loại (Threshold):</label>
                        <span id="thresholdVal" class="text-xs font-mono font-bold text-sky-400 bg-sky-950/60 px-2 py-0.5 rounded border border-sky-800">0.40</span>
                    </div>
                    <input type="range" id="thresholdSlider" min="0.10" max="0.90" step="0.01" value="0.40" 
                           class="w-full h-1.5 bg-slate-700 rounded appearance-none cursor-pointer accent-sky-400"
                           oninput="updateThreshold(this.value)">
                    <div class="flex justify-between text-[11px] text-slate-500 mt-1">
                        <span>0.10</span>
                        <span>0.40 (Ngưỡng demo)</span>
                        <span>0.90</span>
                    </div>
                </div>

                <div>
                    <label for="soupMode" class="text-xs text-slate-300">Chế độ Soup</label>
                    <select id="soupMode" onchange="changeSoupMode()" class="block mt-1 p-2 rounded bg-slate-800 text-slate-100 text-xs border border-slate-600">
                        <option value="raw">Soup gốc</option>
                        <option value="leace">Soup + LEACE</option>
                    </select>
                </div>
                <div>
                    <button id="btnCompare" onclick="runComparison()" 
                            class="px-6 py-2.5 rounded bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs tracking-wide transition flex items-center gap-2 disabled:opacity-50">
                        <span>Tính toán đối chiếu</span>
                    </button>
                </div>
            </div>
            <p class="text-xs text-slate-400 mt-3">LEACE giảm thông tin phong cách trong ID512. Chế độ này đã giúp ảnh phong cách nhưng giảm một phần kết quả ảnh thật. Ngưỡng ở đây để thử nghiệm, chưa hiệu chuẩn riêng cho từng chế độ.</p>
        </div>

        <!-- Loading -->
        <div id="loadingBox" class="hidden text-center py-8">
            <div class="inline-block animate-spin rounded-full h-8 w-8 border-2 border-sky-500 border-t-transparent mb-2"></div>
            <p class="text-xs text-slate-400">Đang trích xuất đặc trưng và tính toán tương đồng...</p>
        </div>

        <!-- Results Container -->
        <div id="resultsContainer" class="hidden space-y-6">

            <!-- Side-by-side Result Cards -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- StyleID Card -->
                <div class="panel rounded-lg p-5">
                    <div class="flex justify-between items-start mb-3">
                        <div>
                            <span class="text-[11px] uppercase font-semibold text-slate-400">Mô hình đối chứng</span>
                            <h3 class="text-sm font-semibold text-slate-100">StyleID (kwanY et al., SIGGRAPH 2026)</h3>
                        </div>
                        <span class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                            768-dim
                        </span>
                    </div>

                    <div class="my-4 p-4 rounded bg-slate-900 border border-slate-800 text-center">
                        <span class="text-xs text-slate-400">Cosine Similarity:</span>
                        <div id="scoreStyleID" class="text-3xl font-mono font-bold text-slate-100 my-1">0.0000</div>
                        <div id="statusBadgeStyleID" class="inline-block mt-2 px-2.5 py-0.5 rounded text-xs font-medium border">
                            —
                        </div>
                    </div>

                    <div class="text-[11px] text-slate-400 space-y-1">
                        <div class="flex justify-between">
                            <span>Thang đo (-1.0 đến +1.0)</span>
                            <span id="barValStyleID">0.0000</span>
                        </div>
                        <div class="w-full bg-slate-800 h-1.5 rounded overflow-hidden">
                            <div id="barStyleID" class="bg-slate-400 h-1.5 rounded transition-all duration-300" style="width: 50%"></div>
                        </div>
                    </div>
                </div>

                <!-- Disentangled-StyleID Card -->
                <div class="panel rounded-lg p-5 border-slate-700">
                    <div class="flex justify-between items-start mb-3">
                        <div>
                            <span class="text-[11px] uppercase font-semibold text-sky-400">Mô hình đề tài</span>
                            <h3 id="soupTitle" class="text-sm font-semibold text-slate-100">Soup gốc</h3>
                        </div>
                        <span class="text-[11px] px-2 py-0.5 rounded bg-sky-950/60 text-sky-400 border border-sky-800 font-medium">
                            ID 512 chiều
                        </span>
                    </div>

                    <div class="my-4 p-4 rounded bg-slate-900 border border-slate-800 text-center">
                        <span class="text-xs text-slate-400">Cosine Similarity:</span>
                        <div id="scoreSoup" class="text-3xl font-mono font-bold text-sky-400 my-1">0.0000</div>
                        <div id="soupBothScores" class="text-xs text-slate-400"></div>
                        <div id="statusBadgeSoup" class="inline-block mt-2 px-2.5 py-0.5 rounded text-xs font-medium border">
                            —
                        </div>
                    </div>

                    <div class="text-[11px] text-slate-400 space-y-1">
                        <div class="flex justify-between">
                            <span>Thang đo (-1.0 đến +1.0)</span>
                            <span id="barValSoup">0.0000</span>
                        </div>
                        <div class="w-full bg-slate-800 h-1.5 rounded overflow-hidden">
                            <div id="barSoup" class="bg-sky-400 h-1.5 rounded transition-all duration-300" style="width: 50%"></div>
                        </div>
                    </div>
                </div>

            </div>

            <!-- Objective Comparison Table -->
            <div class="panel rounded-lg p-4">
                <h4 class="text-xs font-semibold text-slate-300 mb-3">Bảng thông số đối chiếu thực nghiệm</h4>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs text-left text-slate-300">
                        <thead class="text-[11px] uppercase bg-slate-900/60 text-slate-400 border-b border-slate-700">
                            <tr>
                                <th class="px-4 py-2">Chỉ số</th>
                                <th class="px-4 py-2">StyleID (Đối chứng)</th>
                                <th id="soupTableTitle" class="px-4 py-2 text-sky-300">Soup gốc</th>
                                <th class="px-4 py-2 text-right">Chênh lệch (Δ)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800">
                            <tr>
                                <td class="px-4 py-2 text-slate-400">Cosine Similarity</td>
                                <td id="tblSimOff" class="px-4 py-2 font-mono">0.0000</td>
                                <td id="tblSimSoup" class="px-4 py-2 font-mono text-sky-300">0.0000</td>
                                <td id="tblDelta" class="px-4 py-2 text-right font-mono font-medium">0.0000</td>
                            </tr>
                            <tr>
                                <td class="px-4 py-2 text-slate-400">Phán đoán tại ngưỡng <span class="tblThVal">0.40</span></td>
                                <td id="tblPredOff" class="px-4 py-2">—</td>
                                <td id="tblPredSoup" class="px-4 py-2 text-sky-300">—</td>
                                <td id="tblPredMatch" class="px-4 py-2 text-right text-slate-400">—</td>
                            </tr>
                            <tr>
                                <td class="px-4 py-2 text-slate-400">Kích thước vector đặc trưng ID</td>
                                <td class="px-4 py-2">768 chiều (CLIP LoRA)</td>
                                <td class="px-4 py-2 text-sky-300">512 chiều (Nhánh danh tính)</td>
                                <td class="px-4 py-2 text-right text-slate-400">-256 chiều</td>
                            </tr>
                            <tr>
                                <td class="px-4 py-2 text-slate-400">Cơ chế xử lý phong cách</td>
                                <td class="px-4 py-2">Không phân rã (Học gộp)</td>
                                <td id="soupMechanism" class="px-4 py-2 text-sky-300">Nhánh ID của Soup</td>
                                <td class="px-4 py-2 text-right text-slate-400">Giảm ảnh hưởng phong cách</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-800 bg-slate-900 py-3 text-center text-xs text-slate-500">
        Đề tài nghiên cứu: Disentangled-StyleID • Môi trường thực nghiệm
    </footer>

    <!-- Logic Script -->
    <script>
        let img1Data = null;
        let img2Data = null;
        let lastResult = null;

        function setupDropzone(zoneId, fileInputId, imgNum) {
            const zone = document.getElementById(zoneId);
            ['dragenter', 'dragover'].forEach(eName => {
                zone.addEventListener(eName, (e) => {
                    e.preventDefault(); e.stopPropagation();
                    zone.classList.add('dropzone-active');
                });
            });
            ['dragleave', 'drop'].forEach(eName => {
                zone.addEventListener(eName, (e) => {
                    e.preventDefault(); e.stopPropagation();
                    zone.classList.remove('dropzone-active');
                });
            });
            zone.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files && files.length > 0) handleFileSelect(imgNum, files);
            });
        }

        setupDropzone('dropzone1', 'fileInput1', 1);
        setupDropzone('dropzone2', 'fileInput2', 2);

        function handleFileSelect(num, files) {
            if (!files || files.length === 0) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                setImagePreview(num, e.target.result);
            };
            reader.readAsDataURL(files[0]);
        }

        function setImagePreview(num, b64) {
            if (num === 1) {
                img1Data = b64;
                document.getElementById('preview1').src = b64;
                document.getElementById('preview1').classList.remove('hidden');
                document.getElementById('placeholder1').classList.add('hidden');
                document.getElementById('btnClear1').classList.remove('hidden');
            } else {
                img2Data = b64;
                document.getElementById('preview2').src = b64;
                document.getElementById('preview2').classList.remove('hidden');
                document.getElementById('placeholder2').classList.add('hidden');
                document.getElementById('btnClear2').classList.remove('hidden');
            }
        }

        function clearImage(num) {
            if (num === 1) {
                img1Data = null;
                document.getElementById('preview1').src = '';
                document.getElementById('preview1').classList.add('hidden');
                document.getElementById('placeholder1').classList.remove('hidden');
                document.getElementById('btnClear1').classList.add('hidden');
                document.getElementById('fileInput1').value = '';
            } else {
                img2Data = null;
                document.getElementById('preview2').src = '';
                document.getElementById('preview2').classList.add('hidden');
                document.getElementById('placeholder2').classList.remove('hidden');
                document.getElementById('btnClear2').classList.add('hidden');
                document.getElementById('fileInput2').value = '';
            }
        }

        async function loadPreset(presetId) {
            try {
                const res = await fetch(`/api/preset/${presetId}`);
                const data = await res.json();
                if (data.img1 && data.img2) {
                    setImagePreview(1, data.img1);
                    setImagePreview(2, data.img2);
                    document.getElementById('presetDesc').innerText = "Mô tả: " + data.desc;
                    document.getElementById('presetGroundTruth').innerText = "Thực tế: " + data.ground_truth;
                    runComparison();
                }
            } catch (err) {
                alert("Không nạp được preset: " + err);
            }
        }

        function updateThreshold(val) {
            document.getElementById('thresholdVal').innerText = parseFloat(val).toFixed(2);
            document.querySelectorAll('.tblThVal').forEach(el => el.innerText = parseFloat(val).toFixed(2));
            if (lastResult) {
                recalcWithThreshold(parseFloat(val));
            }
        }

        async function runComparison() {
            if (!img1Data || !img2Data) {
                alert("Vui lòng tải lên cả 2 ảnh hoặc chọn một cặp ảnh mẫu.");
                return;
            }

            const th = parseFloat(document.getElementById('thresholdSlider').value);
            const btn = document.getElementById('btnCompare');
            const loading = document.getElementById('loadingBox');
            const results = document.getElementById('resultsContainer');

            btn.disabled = true;
            loading.classList.remove('hidden');
            results.classList.add('hidden');

            try {
                const res = await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        img1_base64: img1Data,
                        img2_base64: img2Data,
                        threshold: th,
                        use_leace: document.getElementById('soupMode').value === 'leace'
                    })
                });
                const data = await res.json();

                if (!res.ok || data.error) {
                    throw new Error(data.error || ("Mã lỗi: " + res.status));
                }
                lastResult = data;
                displayResults(data);
            } catch (err) {
                alert("Lỗi khi xử lý: " + err.message);
            } finally {
                btn.disabled = false;
                loading.classList.add('hidden');
            }
        }

        function displayResults(data) {
            document.getElementById('deviceBadge').innerText = data.device;
            document.getElementById('resultsContainer').classList.remove('hidden');
            recalcWithThreshold(parseFloat(document.getElementById('thresholdSlider').value));
        }

        function recalcWithThreshold(th) {
            if (!lastResult) return;
            const simOff = lastResult.styleid.similarity;
            const useLeace = document.getElementById('soupMode').value === 'leace';
            const simSoup = (useLeace ? lastResult.soup_leace : lastResult.soup_raw).similarity;
            const title = useLeace ? 'Soup + LEACE' : 'Soup gốc';
            document.getElementById('soupTitle').innerText = title;
            document.getElementById('soupTableTitle').innerText = title;
            document.getElementById('soupMechanism').innerText = useLeace ? 'ID512 → LEACE → chuẩn hóa L2' : 'Nhánh ID của Soup';
            document.getElementById('soupBothScores').innerText = `Gốc: ${lastResult.soup_raw.similarity.toFixed(4)} · LEACE: ${lastResult.soup_leace.similarity.toFixed(4)}`;
            const delta = simSoup - simOff;

            const isSameOff = simOff >= th;
            const isSameSoup = simSoup >= th;

            // 1. StyleID Card
            document.getElementById('scoreStyleID').innerText = simOff.toFixed(4);
            const badgeOff = document.getElementById('statusBadgeStyleID');
            badgeOff.innerText = isSameOff ? "Cùng danh tính (s ≥ th)" : "Khác danh tính (s < th)";
            badgeOff.className = `inline-block mt-2 px-2.5 py-0.5 rounded text-xs font-medium border ${isSameOff ? 'bg-sky-950/60 text-sky-400 border-sky-800' : 'bg-slate-800 text-slate-400 border-slate-700'}`;

            const barPctOff = Math.max(0, Math.min(100, ((simOff + 1.0) / 2.0) * 100));
            document.getElementById('barStyleID').style.width = `${barPctOff}%`;
            document.getElementById('barValStyleID').innerText = simOff.toFixed(4);

            // 2. Soup Card
            document.getElementById('scoreSoup').innerText = simSoup.toFixed(4);
            const badgeSoup = document.getElementById('statusBadgeSoup');
            badgeSoup.innerText = isSameSoup ? "Cùng danh tính (s ≥ th)" : "Khác danh tính (s < th)";
            badgeSoup.className = `inline-block mt-2 px-2.5 py-0.5 rounded text-xs font-medium border ${isSameSoup ? 'bg-sky-950/60 text-sky-400 border-sky-800' : 'bg-slate-800 text-slate-400 border-slate-700'}`;

            const barPctSoup = Math.max(0, Math.min(100, ((simSoup + 1.0) / 2.0) * 100));
            document.getElementById('barSoup').style.width = `${barPctSoup}%`;
            document.getElementById('barValSoup').innerText = simSoup.toFixed(4);

            // 3. Breakdown Table
            document.getElementById('tblSimOff').innerText = simOff.toFixed(4);
            document.getElementById('tblSimSoup').innerText = simSoup.toFixed(4);
            document.getElementById('tblDelta').innerText = `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`;

            document.getElementById('tblPredOff').innerText = isSameOff ? "Cùng danh tính" : "Khác danh tính";
            document.getElementById('tblPredSoup').innerText = isSameSoup ? "Cùng danh tính" : "Khác danh tính";

            if (isSameOff === isSameSoup) {
                document.getElementById('tblPredMatch').innerText = "Hai mô hình đồng thuận";
            } else {
                document.getElementById('tblPredMatch').innerText = "Hai mô hình phân kỳ";
            }

            document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function changeSoupMode() {
            const title = document.getElementById('soupMode').value === 'leace' ? 'Soup + LEACE' : 'Soup gốc';
            document.getElementById('soupTitle').innerText = title;
            document.getElementById('soupTableTitle').innerText = title;
            if (lastResult) recalcWithThreshold(parseFloat(document.getElementById('thresholdSlider').value));
        }

        window.addEventListener('DOMContentLoaded', () => {
            loadPreset('1');
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return HTMLResponse(content=HTML_PAGE)

def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == '__main__':
    print("Web demo khoi chay tai: http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
