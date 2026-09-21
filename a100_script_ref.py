 ==============================================================================
# DISENTANGLED-STYLEID: MEGA-BATCH TRAINING TRÊN NVIDIA A100 80GB (1-CLICK RUN)
# ==============================================================================
import os, glob, json, tarfile, shutil, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image
import tqdm

# 1. Kiểm tra phần cứng A100 80GB
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print("="*80)
print(f"🔥 GPU CAO CẤP: {gpu_name.upper()} | VRAM: {vram_gb:.1f} GB")
print(f"🔥 CẤU HÌNH: BATCH SIZE 128 | BFLOAT16 ACCELERATION | COSINE ANNEALING LR")
print("="*80)

# Kết nối Google Drive
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'): drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/StyleID_A100_Champion'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"--> ✅ Thư mục lưu Checkpoint Google Drive: {SAVE_DIR}")
except Exception:
    SAVE_DIR = './'

MEGA_DATA_DIR = "/content/mega_train_data"
os.makedirs(MEGA_DATA_DIR, exist_ok=True)

# 2. Tải và gộp đồng thời các Shard lớn nhất vào chung 1 thư mục
TARGET_SHARDS = [1, 2, 4, 5, 6]  # Gộp ~110.000 ảnh tinh hoa nhất
print(f"\n--> [BƯỚC 1/4] Đang gom đồng thời {len(TARGET_SHARDS)} Shards lớn vào chung 1 thư mục...")

for s in TARGET_SHARDS:
    tar_name = f"styleid-s-{s:06d}.tar"
    print(f"    + Đang kéo {tar_name}...")
    tar_path = hf_hub_download(repo_id="kwanY/stylebench-s", filename=tar_name, repo_type="dataset", local_dir="./")
    
    last_png_bytes = None
    with tarfile.open(tar_path, 'r') as tar:
        for m in tqdm.tqdm(tar, desc=f"    Gộp Shard {s}"):
            if m.name.endswith(".png"):
                f = tar.extractfile(m)
                if f is not None: last_png_bytes = f.read()
            elif m.name.endswith(".json") and last_png_bytes is not None:
                f = tar.extractfile(m)
                if f is not None:
                    try:
                        meta = json.loads(f.read().decode("utf-8"))
                        orig = meta.get("original_path")
                        if orig:
                            dest = os.path.join(MEGA_DATA_DIR, orig)
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            with open(dest, "wb") as out_f: out_f.write(last_png_bytes)
                    except Exception: pass
                last_png_bytes = None
    try: os.remove(tar_path)
    except Exception: pass

total_train_imgs = len(glob.glob(f"{MEGA_DATA_DIR}/**/*.png", recursive=True))
print(f"\n--> ✅ ĐÃ GỘP THÀNH CÔNG: {total_train_imgs:,} BỨC ẢNH VÀO CHUNG 1 TẬP HUẤN LUYỆN!")

# 3. Định nghĩa kiến trúc Disentangled-StyleID
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
    def __init__(self, clip_dim=768, id_dim=512, style_dim=256, num_styles=10):
        super().__init__()
        self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
        self.style_proj = nn.Sequential(nn.Linear(clip_dim, style_dim), nn.BatchNorm1d(style_dim), nn.PReLU())
        self.style_classifier = nn.Linear(style_dim, max(num_styles, 2))
        self.ortho_proj = nn.Linear(id_dim, style_dim, bias=False)
    def forward(self, feat):
        z_id = F.normalize(self.id_proj(feat), dim=-1)
        z_style = F.normalize(self.style_proj(feat), dim=-1)
        return z_id, z_style, self.style_classifier(z_style)
    def compute_ortho_loss(self, z_id, z_style):
        z_id_p = F.normalize(self.ortho_proj(z_id), dim=-1)
        return ((z_id_p * z_style).sum(dim=-1) ** 2).mean()

class AngularHead(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=0.3, scale=30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.margin, self.scale = margin, scale
    def forward(self, x, labels=None):
        logits = F.normalize(x, dim=-1) @ F.normalize(self.W, dim=-1).t()
        if labels is None: return logits
        theta = torch.acos(logits.clamp(-1 + 1e-5, 1 - 1e-5))
        target_logits = torch.cos(theta + self.margin)
        onehot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        return (logits * (1 - onehot) + target_logits * onehot) * self.scale

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        batch_size = features.shape[0]
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)
        mask_pos = torch.where(mask.sum(1) < 1e-6, 1.0, mask.sum(1))
        return -((mask * log_prob).sum(1) / mask_pos).mean()

def get_features(m, pv):
    return m.visual_projection(m.vision_model(pixel_values=pv).pooler_output)

# 4. DataLoader Tốc Độ Cao Cho A100 80GB
class MultiStyleDataset(Dataset):
    def __init__(self, root):
        self.samples, self.class_to_idx, self.style_to_idx = [], {}, {}
        for m in sorted(os.listdir(root)):
            mp = os.path.join(root, m)
            if not os.path.isdir(mp): continue
            for cat in sorted(os.listdir(mp)):
                cp = os.path.join(mp, cat)
                if not os.path.isdir(cp): continue
                for p in glob.glob(os.path.join(cp, "*.png")):
                    ident = os.path.basename(p).split("_")[0]
                    if ident not in self.class_to_idx: self.class_to_idx[ident] = len(self.class_to_idx)
                    if cat not in self.style_to_idx: self.style_to_idx[cat] = len(self.style_to_idx)
                    self.samples.append((p, self.class_to_idx[ident], self.style_to_idx[cat]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, id_lbl, sty_lbl = self.samples[idx]
        return load_image(p).convert("RGB"), id_lbl, sty_lbl

def collate_fn(batch):
    images, id_labels, style_labels = zip(*batch)
    return list(images), torch.tensor(id_labels, dtype=torch.long), torch.tensor(style_labels, dtype=torch.long)

# 5. Huấn luyện toàn bộ dữ liệu (Mega-Batch Training)
print(f"\n--> [BƯỚC 2/4] Khởi tạo mô hình và DataLoader Batch Size 128...")
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
for p in model.parameters(): p.requires_grad = False
apply_lora_to_clip(model, r=8)

head = DisentangledHead(clip_dim=model.config.projection_dim, id_dim=512, style_dim=256).to(device)
ce_loss, supcon_fn = nn.CrossEntropyLoss(), SupConLoss().to(device)

train_dataset = MultiStyleDataset(MEGA_DATA_DIR)
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,  # BATCH SIZE 128 CỰC MẠNH
    collate_fn=collate_fn, drop_last=True, num_workers=8, pin_memory=True
)
angular_head = AngularHead(512, len(train_dataset.class_to_idx)).to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters()) + list(angular_head.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1.2e-4, weight_decay=1e-4)
epochs = 2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)

print(f"\n--> [BƯỚC 3/4] BẮT ĐẦU HUẤN LUYỆN A100 (2 EPOCHS TRÊN TOÀN BỘ 110K ẢNH TRỘN LẪN)...")
for epoch in range(epochs):
    model.train(); head.train()
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (A100)")
    for step, (images, id_labels, style_labels) in enumerate(pbar):
        id_labels, style_labels = id_labels.to(device), style_labels.to(device)
        pv = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
        
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feat = get_features(model, pv)
            z_id, z_style, style_logits = head(feat)
            loss_id = ce_loss(angular_head(z_id, id_labels), id_labels) + supcon_fn(z_id, id_labels)
            loss_style = ce_loss(style_logits, style_labels)
            loss_ortho = head.compute_ortho_loss(z_id, z_style)
            loss = loss_id + 0.5 * loss_style + 0.1 * loss_ortho

        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

# Lưu Checkpoint Đỉnh Cao
torch.save(model.state_dict(), 'disentangled_lora_clip.pt')
torch.save(head.state_dict(), 'disentangled_head.pt')
torch.save(model.state_dict(), f"{SAVE_DIR}/disentangled_lora_clip_a100_final.pt")
torch.save(head.state_dict(), f"{SAVE_DIR}/disentangled_head_a100_final.pt")
print(f"\n--> 💾 ĐÃ LƯU TRỌNG SỐ VÔ ĐỊCH VÀO GOOGLE DRIVE: {SAVE_DIR}!")

# 6. TỰ ĐỘNG CHẤM ĐIỂM SKSF-A BẢNG 2 NGAY LẬP TỨC
print(f"\n--> [BƯỚC 4/4] ĐANG ĐỐI ĐẦU TRỰC TIẾP TRÊN BẢNG 2 (SKSF-A PHOTO-TO-SKETCH)...")
!pip install -q gdown
if not os.path.exists('SKSF-A_data'):
    if not os.path.exists('SKSF-A.zip'):
        !gdown 1E0QBScC_0cXoMNn7JE1P4f6I9qMFagvz -O SKSF-A.zip
    !unzip -q SKSF-A.zip -d SKSF-A_data

def load_image_clean(p):
    img = Image.open(p)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        c = Image.new('RGB', img.size, (255, 255, 255))
        c.paste(img, mask=img.split()[3])
        return c
    return img.convert('RGB')

def find_file(f, idx):
    for ext in ['.PNG', '.png', '.jpg', '.jpeg', '.JPG']:
        p = os.path.join(f, f"{idx}{ext}")
        if os.path.exists(p): return p
    return None

import pandas as pd
df = pd.read_csv('SKSF-A_data/sksf_a.csv')
image_ids = df['Image_no'].tolist()
styles = [f"style{i}" for i in range(1, 8)]

pairs_pos, pairs_neg = [], []
random.seed(42)
for img_id in image_ids:
    photo_p = find_file('SKSF-A_data/Photo', img_id)
    if not photo_p: continue
    for st in styles:
        sketch_p = find_file(f'SKSF-A_data/{st}', img_id)
        if sketch_p:
            pairs_pos.append((photo_p, sketch_p, 1))
            other_ids = [o for o in image_ids if o != img_id]
            neg_p = find_file(f'SKSF-A_data/{st}', random.choice(other_ids))
            if neg_p: pairs_neg.append((photo_p, neg_p, 0))

eval_pairs = pairs_pos + pairs_neg

# Nạp StyleID chính chủ để đối đầu
off_proc = CLIPProcessor.from_pretrained('kwanY/styleid')
off_model = CLIPModel.from_pretrained('kwanY/styleid').to(device).eval()
model.eval(); head.eval()

cache_my, cache_off = {}, {}
def get_emb_my(p):
    if p not in cache_my:
        pv = processor(images=load_image_clean(p), return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad(): cache_my[p] = head(get_features(model, pv))[0]
    return cache_my[p]

def get_emb_off(p):
    if p not in cache_off:
        pv = off_proc(images=load_image_clean(p), return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad(): cache_off[p] = F.normalize(get_features(off_model, pv), dim=-1)
    return cache_off[p]

y_true, sc_my, sc_off = [], [], []
for p1, p2, lbl in tqdm.tqdm(eval_pairs, desc="Chấm điểm SKSF-A"):
    sc_my.append((get_emb_my(p1) @ get_emb_my(p2).T).item())
    sc_off.append((get_emb_off(p1) @ get_emb_off(p2).T).item())
    y_true.append(lbl)

y_true, sc_my, sc_off = np.array(y_true), np.array(sc_my), np.array(sc_off)

def get_metrics(y, s):
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    tpr1 = tpr[np.where(fpr <= 0.01)[0][-1]]
    pm, nm = (y == 1), (y == 0)
    a3 = (np.sum(s[pm] >= 0.3) + np.sum(s[nm] < 0.3)) / len(y)
    a4 = (np.sum(s[pm] >= 0.4) + np.sum(s[nm] < 0.4)) / len(y)
    a5 = (np.sum(s[pm] >= 0.5) + np.sum(s[nm] < 0.5)) / len(y)
    return tpr1, a3, a4, a5, auc, np.mean(s[pm]), np.mean(s[nm])

tpr_m, a3_m, a4_m, a5_m, auc_m, p_m, n_m = get_metrics(y_true, sc_my)
tpr_o, a3_o, a4_o, a5_o, auc_o, p_o, n_o = get_metrics(y_true, sc_off)

print('\n' + '='*85)
print('BẢNG ĐỐI ĐẦU CHÍNH THỨC SKSF-A (TABLE 2) - KỶ LỤC MỚI CỦA NVIDIA A100 80GB')
print('='*85)
print(f"{'Methods':<28} | {'TPR↑':<8} | {'Acc@0.3↑':<9} | {'Acc@0.4↑':<9} | {'Acc@0.5↑':<9} | {'AUROC↑':<8}")
print('-'*85)
print(f"{'ArcFace (Bài báo công bố)':<28} | 0.6189   | 0.6801    | 0.5346    | 0.5059    | 0.9360")
print(f"{'AdaFace (Bài báo công bố)':<28} | 0.6993   | 0.6178    | 0.5357    | 0.5032    | 0.9582")
print(f"{'CLIP (Bài báo công bố)':<28} | 0.4968   | 0.5000    | 0.5059    | 0.5778    | 0.9420")
print(f"{'StyleID (Bài báo công bố)':<28} | 0.8891   | 0.8731    | 0.7393    | 0.6178    | 0.9922")
print('-'*85)
print(f"{'StyleID (Chính chủ test lại)':<28} | {tpr_o:<8.4f} | {a3_o:<9.4f} | {a4_o:<9.4f} | {a5_o:<9.4f} | {auc_o:<8.4f}")
print(f"{'Disentangled (Bản A100 Mới)':<28} | {tpr_m:<8.4f} | {a3_m:<9.4f} | {a4_m:<9.4f} | {a5_m:<9.4f} | {auc_m:<8.4f}")
print('='*85)
print(f"- Cùng người (Pos Sim) ↑: Bản của bạn: {p_m:.4f} | StyleID chính chủ: {p_o:.4f}")
print(f"- Khoảng cách biên (Margin) ↑: Bản của bạn: {p_m - n_m:.4f} | StyleID chính chủ: {p_o - n_o:.4f}")
