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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
STATIC_DIR = Path(__file__).resolve().with_name('static')
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
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


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


@app.get("/", response_class=FileResponse)
async def serve_home():
    return FileResponse(STATIC_DIR / 'index.html')

def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == '__main__':
    print("Web demo khoi chay tai: http://localhost:8000")
    Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
