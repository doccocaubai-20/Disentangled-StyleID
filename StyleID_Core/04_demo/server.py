"""Compare StyleID with the final Dual-Branch configurations."""
import base64
from binascii import Error as Base64Error
from functools import lru_cache
import io
from pathlib import Path
from threading import Lock, Thread
from typing import Literal
import webbrowser

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor
import uvicorn

from checkpoints import SOUP_SHA256, adapter_path, soup_paths
from demo_hard_positive import load_hard_positive
from lfw_presets import PRESETS, preset_paths


load_dotenv(Path(__file__).resolve().with_name('.env'), override=False)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'
if DEVICE == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

BASE_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
STYLEID_REVISION = '1967c354f339a636e5b3e16ecab3d0075aa27ab1'
STYLEID_MODEL = 'kwanY/styleid'
MODEL_LOCK = Lock()
MODEL_MODES = {
    'dual_hp': {
        'name': 'Dual-Branch + Hard-Positive',
        'dimension': 1280,
        'mechanism': '50% StyleID + 50% ID512 sau Hard-Positive',
    },
    'dual': {
        'name': 'Dual-Branch',
        'dimension': 1280,
        'mechanism': '50% StyleID + 50% ID512',
    },
    'raw': {
        'name': 'Disentangle ID512',
        'dimension': 512,
        'mechanism': 'Nhánh danh tính ID512',
    },
}


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
        attn.q_proj = LoRALinear(attn.q_proj, r=r)
        attn.k_proj = LoRALinear(attn.k_proj, r=r)
        attn.v_proj = LoRALinear(attn.v_proj, r=r)
        attn.out_proj = LoRALinear(attn.out_proj, r=r)

class DisentangledHead(nn.Module):
    def __init__(self, clip_dim=768, id_dim=512):
        super().__init__()
        self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
    def forward(self, feat):
        return F.normalize(self.id_proj(feat), dim=-1)

def load_image_clean(img):
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        canvas = Image.new('RGB', rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel('A'))
        return canvas
    return img.convert('RGB')


@lru_cache(maxsize=1)
def load_styleid():
    processor = CLIPProcessor.from_pretrained(STYLEID_MODEL, revision=STYLEID_REVISION)
    model = CLIPModel.from_pretrained(
        STYLEID_MODEL, revision=STYLEID_REVISION, torch_dtype=torch.float32, attn_implementation='eager',
    )
    return processor, model.requires_grad_(False).eval().to(DEVICE)


@lru_cache(maxsize=1)
def load_soup():
    backbone_path, head_path = soup_paths()
    model = CLIPModel.from_pretrained(
        'openai/clip-vit-large-patch14', revision=BASE_REVISION,
        torch_dtype=torch.float32, attn_implementation='eager',
    )
    apply_lora_to_clip(model)
    model.load_state_dict(torch.load(backbone_path, map_location='cpu', weights_only=True, mmap=True))
    head = DisentangledHead()
    head_state = torch.load(head_path, map_location='cpu', weights_only=True)
    head.load_state_dict({key: value for key, value in head_state.items() if key.startswith('id_proj.')})
    return model.requires_grad_(False).eval().to(DEVICE), head.requires_grad_(False).eval().to(DEVICE)


@lru_cache(maxsize=1)
def load_adapter():
    return load_hard_positive(
        adapter_path(), DEVICE, soup_sha256=SOUP_SHA256,
        base_revision=BASE_REVISION, styleid_revision=STYLEID_REVISION,
    )


def clip_features(model, pixels):
    return model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)


@torch.inference_mode()
def score_pair(images, mode):
    # Serialize first loads and inference so simultaneous requests share the same cached models.
    with MODEL_LOCK:
        adapter = load_adapter() if mode == 'dual_hp' else None
        processor, official = load_styleid()
        soup, head = load_soup()
        pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(DEVICE)
        official_vectors = F.normalize(clip_features(official, pixels), dim=-1).double()
        identity_vectors = head(clip_features(soup, pixels))
        styleid = (official_vectors[0] * official_vectors[1]).sum().item()
        raw = (identity_vectors[0].double() * identity_vectors[1].double()).sum().item()
        scores = {'raw': raw, 'dual': 0.5 * (styleid + raw)}
        adapted = None
        if adapter is not None:
            adapted_vectors = adapter(identity_vectors.float()).double()
            adapted = (adapted_vectors[0] * adapted_vectors[1]).sum().item()
            scores['dual_hp'] = 0.5 * (styleid + adapted)
        return styleid, scores, adapted


app = FastAPI(title='Face Verification Benchmark')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class CompareRequest(BaseModel):
    img1_base64: str | None = None
    img2_base64: str | None = None
    preset_id: str | None = None
    threshold: float = Field(0.30, ge=-1, le=1)
    mode: Literal['dual_hp', 'dual', 'raw', 'leace'] | None = None
    use_leace: bool | None = None


def get_preset_paths(preset_id):
    if preset_id not in PRESETS:
        raise HTTPException(404, 'Không tìm thấy cặp ảnh LFW')
    try:
        return preset_paths(preset_id)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, f'Không lấy được ảnh mẫu LFW: {exc}') from exc


@app.get('/api/preset/{preset_id}')
def get_preset(preset_id: str):
    left, right = get_preset_paths(preset_id)
    title, same_identity, _ = PRESETS[preset_id]
    return {
        'id': preset_id,
        'title': title,
        'desc': f'LFW View 2 · {preset_id} · {title}',
        'ground_truth': 'Cùng danh tính' if same_identity else 'Khác danh tính',
        'img1': 'data:image/jpeg;base64,' + base64.b64encode(left.read_bytes()).decode('ascii'),
        'img2': 'data:image/jpeg;base64,' + base64.b64encode(right.read_bytes()).decode('ascii'),
    }


@app.post('/api/compare')
def compare_images(req: CompareRequest):
    if req.preset_id is not None and req.preset_id not in PRESETS:
        raise HTTPException(404, 'Không tìm thấy cặp ảnh LFW')
    mode = req.mode
    if mode is None:
        mode = {True: 'leace', False: 'raw', None: 'dual_hp' if req.preset_id else 'dual'}[req.use_leace]
    if mode == 'leace':
        raise HTTPException(422, 'Chế độ LEACE đã được gỡ khỏi demo')
    if mode == 'dual_hp' and not req.preset_id:
        raise HTTPException(422, 'Hard-Positive chỉ dùng cho các cặp mẫu LFW')

    if req.preset_id:
        inputs = get_preset_paths(req.preset_id)
    else:
        if not req.img1_base64 or not req.img2_base64:
            raise HTTPException(400, 'Vui lòng chọn đủ hai ảnh hoặc một cặp mẫu LFW')
        try:
            inputs = [
                io.BytesIO(base64.b64decode(data.split(',')[-1], validate=True))
                for data in (req.img1_base64, req.img2_base64)
            ]
        except Base64Error as exc:
            raise HTTPException(400, 'Dữ liệu ảnh không hợp lệ') from exc
    try:
        images = []
        for item in inputs:
            with Image.open(item) as image:
                images.append(load_image_clean(image))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, 'Không đọc được ảnh') from exc

    try:
        styleid, scores, adapted = score_pair(images, mode)
    except (OSError, RuntimeError, ValueError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=503)

    def result(score, dimension=512):
        same = score >= req.threshold
        return {
            'similarity': score,
            'is_same': same,
            'label': 'Cùng danh tính' if same else 'Khác danh tính',
            'dim': f'{dimension} chiều',
        }

    modes = {
        key: {**result(score, MODEL_MODES[key]['dimension']), **MODEL_MODES[key]}
        for key, score in scores.items()
    }
    response = {
        'threshold': req.threshold,
        'device': f'{DEVICE.upper()} ({DEVICE_NAME})',
        'model_name': modes[mode]['name'],
        'mode': mode,
        'use_leace': False,
        'modes': modes,
        'styleid': result(styleid, 768),
        'soup_raw': modes['raw'],
        'dual_branch': modes['dual'],
        'soup': modes[mode],
        'comparison': {'delta': round(scores[mode] - styleid, 4)},
    }
    if adapted is not None:
        response['id_hard_positive'] = result(adapted)
        response['dual_hard_positive'] = modes['dual_hp']
    return response


HTML_PAGE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StyleID và Dual-Branch + Hard-Positive</title>
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
                    StyleID và Dual-Branch + Hard-Positive
                </h1>
                <p class="text-xs text-slate-400">Demo cấu hình cuối LFW từ notebook kiểm chứng Colab</p>
            </div>
            <div class="flex items-center gap-2 text-xs">
                <span class="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300">
                    Mô hình: <span id="activeModelBadge" class="font-medium text-slate-100">Dual-Branch</span>
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
                    Cặp ảnh mẫu LFW View 2
                </div>
                <div class="flex flex-wrap gap-2">
                    <button onclick="loadPreset('lfw-1342')" class="preset-button px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition">
                        Cùng người · một ảnh đeo kính râm
                    </button>
                    <button onclick="loadPreset('lfw-0557')" class="preset-button px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition">
                        Khác người · cặp khó
                    </button>
                    <button onclick="loadPreset('lfw-0000')" class="preset-button px-3 py-1.5 text-xs rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition">
                        Cùng người · cặp đối chứng
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
                    <span class="text-xs font-semibold text-slate-300">Ảnh 2: Khuôn mặt đối chiếu</span>
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
                        <span id="thresholdVal" class="text-xs font-mono font-bold text-sky-400 bg-sky-950/60 px-2 py-0.5 rounded border border-sky-800">0.30</span>
                    </div>
                    <input type="range" id="thresholdSlider" min="0.10" max="0.90" step="0.01" value="0.30"
                           class="w-full h-1.5 bg-slate-700 rounded appearance-none cursor-pointer accent-sky-400"
                           oninput="updateThreshold(this.value)">
                    <div class="flex justify-between text-[11px] text-slate-500 mt-1">
                        <span>0.10</span>
                        <span>0.30 (Ngưỡng demo)</span>
                        <span>0.90</span>
                    </div>
                </div>

                <div>
                    <label for="soupMode" class="text-xs text-slate-300">Cấu hình so sánh</label>
                    <select id="soupMode" onchange="changeSoupMode()" class="block mt-1 p-2 rounded bg-slate-800 text-slate-100 text-xs border border-slate-600">
                        <option value="dual_hp" disabled>Dual-Branch + Hard-Positive</option>
                        <option value="dual" selected>Dual-Branch</option>
                        <option value="raw">Disentangle ID512</option>
                    </select>
                </div>
                <div>
                    <button id="btnCompare" onclick="runComparison()" 
                            class="px-6 py-2.5 rounded bg-sky-600 hover:bg-sky-500 text-white font-medium text-xs tracking-wide transition flex items-center gap-2 disabled:opacity-50">
                        <span>Tính toán đối chiếu</span>
                    </button>
                </div>
            </div>
            <p class="text-xs text-slate-400 mt-3">Chọn một cặp LFW để so sánh với Dual-Branch + Hard-Positive. Ngưỡng mặc định 0,30 tương ứng cột Acc@0.3 trong bảng kiểm chứng.</p>
        </div>

        <!-- Loading -->
        <div id="loadingBox" class="hidden text-center py-8">
            <div class="inline-block animate-spin rounded-full h-8 w-8 border-2 border-sky-500 border-t-transparent mb-2"></div>
            <p class="text-xs text-slate-400">Đang nạp dữ liệu hoặc tính điểm tương đồng…</p>
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
                            <h3 id="soupTitle" class="text-sm font-semibold text-slate-100">Dual-Branch + Hard-Positive</h3>
                        </div>
                        <span id="soupDimensionBadge" class="text-[11px] px-2 py-0.5 rounded bg-sky-950/60 text-sky-400 border border-sky-800 font-medium">
                            1.280 chiều
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
                                <th id="soupTableTitle" class="px-4 py-2 text-sky-300">Dual-Branch + Hard-Positive</th>
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
                                <td class="px-4 py-2 text-slate-400">Phán đoán tại ngưỡng <span class="tblThVal">0.30</span></td>
                                <td id="tblPredOff" class="px-4 py-2">—</td>
                                <td id="tblPredSoup" class="px-4 py-2 text-sky-300">—</td>
                                <td id="tblPredMatch" class="px-4 py-2 text-right text-slate-400">—</td>
                            </tr>
                            <tr>
                                <td class="px-4 py-2 text-slate-400">Kích thước vector so sánh</td>
                                <td class="px-4 py-2">768 chiều (StyleID)</td>
                                <td id="soupDimension" class="px-4 py-2 text-sky-300">1.280 chiều</td>
                                <td id="dimensionDelta" class="px-4 py-2 text-right text-slate-400">+512 chiều</td>
                            </tr>
                            <tr>
                                <td class="px-4 py-2 text-slate-400">Cách tính biểu diễn</td>
                                <td class="px-4 py-2">Không phân rã (Học gộp)</td>
                                <td id="soupMechanism" class="px-4 py-2 text-sky-300">50% StyleID + 50% ID512 sau Hard-Positive</td>
                                <td id="soupMechanismNote" class="px-4 py-2 text-right text-slate-400">Kết hợp hai nhánh</td>
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
        let activePreset = null;
        let inputVersion = 0;
        let busy = false;

        function setBusy(value) {
            busy = value;
            document.querySelectorAll('.preset-button, #btnCompare, #soupMode').forEach(control => {
                control.disabled = value;
            });
            document.getElementById('loadingBox').classList.toggle('hidden', !value);
        }

        function invalidateResult() {
            lastResult = null;
            document.getElementById('resultsContainer').classList.add('hidden');
        }

        function useCustomImages() {
            inputVersion++;
            activePreset = null;
            invalidateResult();
            document.getElementById('presetDesc').innerText = 'Ảnh tải lên';
            document.getElementById('presetGroundTruth').innerText = '';
            const selector = document.getElementById('soupMode');
            selector.querySelector('[value="dual_hp"]').disabled = true;
            if (selector.value === 'dual_hp') selector.value = 'dual';
            changeSoupMode();
        }

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
                useCustomImages();
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
            useCustomImages();
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
            if (busy) return;
            const version = ++inputVersion;
            setBusy(true);
            try {
                const res = await fetch(`/api/preset/${presetId}`);
                const data = await res.json();
                if (version !== inputVersion) return;
                if (!res.ok) throw new Error(data.detail || res.status);
                invalidateResult();
                setImagePreview(1, data.img1);
                setImagePreview(2, data.img2);
                activePreset = presetId;
                document.getElementById('presetDesc').innerText = data.desc;
                document.getElementById('presetGroundTruth').innerText = data.ground_truth;
                const selector = document.getElementById('soupMode');
                selector.querySelector('[value="dual_hp"]').disabled = false;
                selector.value = 'dual_hp';
                changeSoupMode();
            } catch (err) {
                alert("Không nạp được cặp LFW: " + err.message);
                return;
            } finally {
                setBusy(false);
            }
            runComparison();
        }

        function updateThreshold(val) {
            document.getElementById('thresholdVal').innerText = parseFloat(val).toFixed(2);
            document.querySelectorAll('.tblThVal').forEach(el => el.innerText = parseFloat(val).toFixed(2));
            if (lastResult) {
                recalcWithThreshold(parseFloat(val));
            }
        }

        async function runComparison() {
            if (busy) return;
            if (!img1Data || !img2Data) {
                alert("Vui lòng tải lên cả 2 ảnh hoặc chọn một cặp ảnh mẫu.");
                return;
            }

            const th = parseFloat(document.getElementById('thresholdSlider').value);
            const version = inputVersion;
            const images = activePreset ? {preset_id: activePreset}
                : {img1_base64: img1Data, img2_base64: img2Data};
            setBusy(true);
            invalidateResult();

            try {
                const res = await fetch('/api/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ...images,
                        threshold: th,
                        mode: document.getElementById('soupMode').value
                    })
                });
                const data = await res.json();
                if (version !== inputVersion) return;

                if (!res.ok || data.error) {
                    throw new Error(data.detail || data.error || ("Mã lỗi: " + res.status));
                }
                lastResult = data;
                displayResults(data);
            } catch (err) {
                alert("Lỗi khi xử lý: " + err.message);
            } finally {
                setBusy(false);
            }
        }

        function displayResults(data) {
            document.getElementById('deviceBadge').innerText = data.device;
            document.getElementById('resultsContainer').classList.remove('hidden');
            recalcWithThreshold(parseFloat(document.getElementById('thresholdSlider').value));
        }

        function recalcWithThreshold(th) {
            const mode = document.getElementById('soupMode').value;
            const selected = lastResult?.modes[mode];
            if (!selected) return;
            const simOff = lastResult.styleid.similarity;
            const simSoup = selected.similarity;
            const title = selected.name;
            document.getElementById('soupTitle').innerText = title;
            document.getElementById('soupTableTitle').innerText = title;
            document.getElementById('activeModelBadge').innerText = title;
            document.getElementById('soupMechanism').innerText = selected.mechanism;
            document.getElementById('soupDimension').innerText = selected.dim;
            document.getElementById('soupDimensionBadge').innerText = selected.dim;
            const dimensionDelta = selected.dimension - 768;
            document.getElementById('dimensionDelta').innerText = `${dimensionDelta >= 0 ? '+' : ''}${dimensionDelta} chiều`;
            document.getElementById('soupMechanismNote').innerText = selected.dimension === 1280 ? 'Kết hợp hai nhánh' : 'Nhánh danh tính';
            const component = mode === 'dual_hp' ? lastResult.id_hard_positive : lastResult.soup_raw;
            document.getElementById('soupBothScores').innerText = selected.dimension === 1280
                ? `StyleID: ${simOff.toFixed(4)} · ID${mode === 'dual_hp' ? ' + HP' : ''}: ${component.similarity.toFixed(4)} · 50/50`
                : `ID512: ${lastResult.soup_raw.similarity.toFixed(4)}`;
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
            const selector = document.getElementById('soupMode');
            const title = selector.options[selector.selectedIndex].text;
            document.getElementById('activeModelBadge').innerText = title;
            document.getElementById('soupTitle').innerText = title;
            document.getElementById('soupTableTitle').innerText = title;
            if (!lastResult) return;
            if (lastResult.modes[selector.value]) {
                recalcWithThreshold(parseFloat(document.getElementById('thresholdSlider').value));
            } else {
                runComparison();
            }
        }
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
    Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
