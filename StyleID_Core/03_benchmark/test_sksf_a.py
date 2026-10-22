# Paths work independently of the terminal's current directory.
from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parents[1]
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_HERE / '.env', override=False)
_DEFAULT_RESOURCES = _HERE if (_HERE/'SKSF-A_data').is_dir() else _HERE.parent
RESOURCE_ROOT = _Path(_os.environ.get('STYLEID_RESOURCES', str(_DEFAULT_RESOURCES))).resolve()
_CKPT_OPTIONS = [_HERE/'checkpoints'/'soup', _HERE, RESOURCE_ROOT]
_DEFAULT_CKPT = next((p for p in _CKPT_OPTIONS if (p/'disentangled_lora_clip_SOUP_S6_A100.pt').is_file()), RESOURCE_ROOT)
CHECKPOINT_ROOT = _Path(_os.environ.get('STYLEID_CHECKPOINT_DIR', str(_DEFAULT_CKPT))).resolve()
_os.chdir(RESOURCE_ROOT)

# ==============================================================================
# BẢNG 2: ĐỐI ĐẦU TRÊN SKSF-A (TABLE 2 - STYLIZED FACE VERIFICATION)
# Giao thức: 1.876 cặp ảnh (938 Cặp Cùng người + 938 Cặp Khác người, 7 Styles vẽ)
# Chạy mượt mà trên Google Colab GPU và Local máy cá nhân
# ==============================================================================
import os
import sys
import glob
import random
import zipfile
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score

# 1. THIẾT BỊ TÍNH TOÁN
try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("--> Đang cài đặt transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
    from transformers import CLIPModel, CLIPProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f"--> [1/4] Thiết bị tính toán: {device.upper()} ({gpu_name})")

# 2. TỰ ĐỘNG CHUẨN BỊ TẬP DỮ LIỆU SKSF-A
print("\n--> [2/4] Kiểm tra dữ liệu kiểm thử SKSF-A...")

# Tìm kiếm thư mục SKSF-A hoặc file zip
sksf_dir = None
for candidate in ['SKSF-A_data', 'SKSF-A', '.']:
    for root, _, files in os.walk(candidate):
        if 'sksf_a.csv' in files:
            sksf_dir = root
            break
    if sksf_dir: break

if not sksf_dir:
    # Tìm file zip
    zip_candidates = ['SKSF-A.zip', '/content/drive/MyDrive/StyleID_A100_Champion/SKSF-A.zip', '/content/drive/MyDrive/SKSF-A.zip']
    zip_path = None
    for z in zip_candidates:
        if os.path.exists(z):
            zip_path = z
            break
    
    if not zip_path:
        print("  + Đang tải SKSF-A.zip qua gdown...")
        os.system("pip install -q gdown")
        os.system("gdown 1E0QBScC_0cXoMNn7JE1P4f6I9qMFagvz -O SKSF-A.zip")
        zip_path = 'SKSF-A.zip'
        
    print(f"  + Đang giải nén {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall('SKSF-A_data')
        
    for root, _, files in os.walk('SKSF-A_data'):
        if 'sksf_a.csv' in files:
            sksf_dir = root
            break

csv_path = os.path.join(sksf_dir, 'sksf_a.csv')
print(f"  + Thư mục SKSF-A: {sksf_dir}")

def find_file(folder, idx):
    for ext in ['.PNG', '.png', '.jpg', '.jpeg', '.JPG']:
        p = os.path.join(folder, f"{idx}{ext}")
        if os.path.exists(p): return p
    return None

df = pd.read_csv(csv_path)
image_ids = df['Image_no'].tolist()
styles = [f"style{i}" for i in range(1, 8)]

pairs_pos, pairs_neg = [], []
random.seed(42)
for img_id in image_ids:
    photo_p = find_file(os.path.join(sksf_dir, 'Photo'), img_id)
    if not photo_p: continue
    for st in styles:
        sk_p = find_file(os.path.join(sksf_dir, st), img_id)
        if sk_p:
            pairs_pos.append((photo_p, sk_p, 1))
            other = random.choice([o for o in image_ids if o != img_id])
            neg_sk_p = find_file(os.path.join(sksf_dir, st), other)
            if neg_sk_p: pairs_neg.append((photo_p, neg_sk_p, 0))

sksf_pairs = pairs_pos + pairs_neg
if len(sksf_pairs) != 1876 or len(pairs_pos) != 938 or len(pairs_neg) != 938:
    raise ValueError('Incomplete SKSF-A: expected 1876 pairs (938 positive, 938 negative)')
y_true = np.array([p[2] for p in sksf_pairs])
unique_paths = sorted(list(set([p[0] for p in sksf_pairs] + [p[1] for p in sksf_pairs])))
print(f"  + Đã nạp: {len(sksf_pairs)} cặp test ({len(pairs_pos)} Cùng danh tính / {len(pairs_neg)} Khác danh tính)")
print(f"  + Tổng số ảnh thực tế: {len(unique_paths)} ảnh")

# 3. TRÍCH XUẤT ĐẶC TRƯNG SIÊU TỐC VỚI BATCH CACHING
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

def load_image_clean(p):
    img = Image.open(p)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        c = Image.new('RGB', img.size, (255, 255, 255))
        c.paste(img, mask=img.split()[3])
        return c
    return img.convert('RGB')

def extract_sksf_embeddings(model, head=None, is_official=True, batch_size=128):
    model.eval()
    if head is not None: head.eval()
    emb_dict = {}
    with torch.no_grad():
        for i in range(0, len(unique_paths), batch_size):
            bp = unique_paths[i:i+batch_size]
            b_imgs = [load_image_clean(p) for p in bp]
            pv = processor(images=b_imgs, return_tensors='pt')['pixel_values'].to(device)
            feats = model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)
            embs = F.normalize(feats, dim=-1) if is_official else head(feats)
            for p, emb in zip(bp, embs.cpu().numpy()):
                emb_dict[p] = emb
                
    sims = [float(np.dot(emb_dict[p1], emb_dict[p2])) for p1, p2, _ in sksf_pairs]
    return np.array(sims)

# 4. HÀM TÍNH CHỈ SỐ TABLE 2
def calc_sksf_metrics(y, s):
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    idx_1fpr = np.where(fpr <= 0.01)[0]
    tpr1 = tpr[idx_1fpr[-1]] if len(idx_1fpr) > 0 else 0.0
    
    pm, nm = (y == 1), (y == 0)
    a3 = (np.sum(s[pm] >= 0.3) + np.sum(s[nm] < 0.3)) / len(y)
    a4 = (np.sum(s[pm] >= 0.4) + np.sum(s[nm] < 0.4)) / len(y)
    a5 = (np.sum(s[pm] >= 0.5) + np.sum(s[nm] < 0.5)) / len(y)
    
    pos_sim = np.mean(s[pm])
    neg_sim = np.mean(s[nm])
    margin = pos_sim - neg_sim
    return {
        'tpr1': tpr1, 'a3': a3, 'a4': a4, 'a5': a5,
        'auc': auc, 'pos_sim': pos_sim, 'neg_sim': neg_sim, 'margin': margin
    }

# 5. CHẤM ĐIỂM STYLEID CHÍNH CHỦ
print("\n--> [3/4] Đang nạp StyleID Chính chủ từ Hugging Face (kwanY/styleid)...")
styleid_model = CLIPModel.from_pretrained("kwanY/styleid").to(device)
scores_styleid = extract_sksf_embeddings(styleid_model, is_official=True, batch_size=128)
metrics_styleid = calc_sksf_metrics(y_true, scores_styleid)
del styleid_model
torch.cuda.empty_cache()

# 6. CHẤM ĐIỂM DISENTANGLED-STYLEID (MODEL SOUP)
soup_ckpt_l, soup_ckpt_h = None, None
for d in [str(CHECKPOINT_ROOT), '/content/drive/MyDrive/StyleID_A100_Champion', '/content/drive/MyDrive', '.']:
    p_l = os.path.join(d, 'disentangled_lora_clip_SOUP_S6_A100.pt')
    p_h = os.path.join(d, 'disentangled_head_SOUP_S6_A100.pt')
    if os.path.exists(p_l) and os.path.exists(p_h):
        soup_ckpt_l, soup_ckpt_h = p_l, p_h
        break

if not soup_ckpt_l or not soup_ckpt_h:
    raise FileNotFoundError('Missing exact Soup backbone/head. Set STYLEID_CHECKPOINT_DIR; no shard fallback.')

metrics_soup = None
if soup_ckpt_l and soup_ckpt_h:
    print(f"\n--> [4/4] Phát hiện mô hình Disentangled tại: {soup_ckpt_l}")
    class LoRALinear(nn.Module):
        def __init__(self, linear_layer, r=8, alpha=1.0):
            super().__init__()
            self.linear, self.scaling = linear_layer, alpha / r
            self.lora_down = nn.Linear(linear_layer.in_features, r, bias=False)
            self.lora_up = nn.Linear(r, linear_layer.out_features, bias=False)
        def forward(self, x): return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling

    class DisentangledHead(nn.Module):
        def __init__(self, clip_dim=768, id_dim=512):
            super().__init__()
            self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
        def forward(self, feat): return F.normalize(self.id_proj(feat), dim=-1)

    soup_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    for p in soup_model.parameters(): p.requires_grad = False
    for layer in soup_model.vision_model.encoder.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=8).to(device)
        layer.self_attn.k_proj = LoRALinear(layer.self_attn.k_proj, r=8).to(device)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=8).to(device)
        layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, r=8).to(device)
        
    soup_head = DisentangledHead().to(device)
    soup_model.load_state_dict(torch.load(soup_ckpt_l, map_location=device), strict=False)
    # nạp với strict=False để bỏ qua các key nhánh style
    soup_head.load_state_dict(torch.load(soup_ckpt_h, map_location=device), strict=False)
    
    scores_soup = extract_sksf_embeddings(soup_model, soup_head, is_official=False, batch_size=128)
    metrics_soup = calc_sksf_metrics(y_true, scores_soup)
    del soup_model, soup_head
    torch.cuda.empty_cache()

# 7. IN BẢNG ĐỐI CHIẾU CHUẨN TABLE 2
print("\n" + "="*100)
print("BẢNG ĐỐI CHIẾU KẾT QUẢ THỰC NGHIỆM TRÊN SKSF-A (TABLE 2 - 1.876 CẶP TRANH NGHỆ THUẬT)")
print("="*100)
print(f"{'Phương pháp':<32} | {'TPR↑ (@1% FPR)':<16} | {'Acc@0.3↑':<10} | {'Acc@0.4↑':<10} | {'Acc@0.5↑':<10} | {'AUROC↑':<10}")
print("-" * 100)
print(f"{'ArcFace (Bài báo công bố)':<32} | 0.6189           | 0.6801     | 0.5346     | 0.5059     | 0.9360")
print(f"{'AdaFace (Bài báo công bố)':<32} | 0.6993           | 0.6178     | 0.5357     | 0.5032     | 0.9582")
print(f"{'CLIP (Bài báo công bố)':<32} | 0.4968           | 0.5000     | 0.5059     | 0.5778     | 0.9420")
print(f"{'StyleID (Bài báo công bố)':<32} | 0.8891           | 0.8731     | 0.7393     | 0.6178     | 0.9922")
print("-" * 100)
print(f"{'StyleID (Chính chủ test lại)':<32} | {metrics_styleid['tpr1']:<16.4f} | {metrics_styleid['a3']:<10.4f} | {metrics_styleid['a4']:<10.4f} | {metrics_styleid['a5']:<10.4f} | {metrics_styleid['auc']:<10.4f}")

if metrics_soup:
    print(f"{'Disentangled (Model Soup)':<32} | {metrics_soup['tpr1']:<16.4f} | {metrics_soup['a3']:<10.4f} | {metrics_soup['a4']:<10.4f} | {metrics_soup['a5']:<10.4f} | {metrics_soup['auc']:<10.4f}")

print("=" * 100)
print("\nCHI TIẾT PHÂN PHỐI ĐỘ TƯƠNG ĐỒNG:")
print(f"  + StyleID Chính chủ       : Pos Sim = {metrics_styleid['pos_sim']:.4f} | Neg Sim = {metrics_styleid['neg_sim']:.4f} | Margin = {metrics_styleid['margin']:.4f}")
if metrics_soup:
    print(f"  + Disentangled (Model Soup): Pos Sim = {metrics_soup['pos_sim']:.4f} | Neg Sim = {metrics_soup['neg_sim']:.4f} | Margin = {metrics_soup['margin']:.4f}")
print("=" * 100)
