# ==============================================================================
# DISENTANGLED-STYLEID: MULTI-SHARD MEGA-BATCH TRAINING (NVIDIA A100)
# Ho tro huan luyen dong thoi nhieu Shards cung luc (Gom data & Shuffle 100%)
# ==============================================================================
import os, glob, json, tarfile, shutil, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image
import tqdm

# 1. CAU HINH THIET BI & THU MUC
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print('='*80)
    print(f'GPU: {gpu_name.upper()} | VRAM: {vram_gb:.1f} GB')
    print('CAU HINH: BATCH SIZE 128 | BFLOAT16 ACCELERATION | COSINE LR SCHEDULER')
    print('='*80)

# Ket noi Google Drive
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'): drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/StyleID_A100_Champion'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'--> Checkpoint Google Drive: {SAVE_DIR}')
except Exception:
    SAVE_DIR = './checkpoints'
    os.makedirs(SAVE_DIR, exist_ok=True)

MEGA_DATA_DIR = '/content/mega_train_data'
os.makedirs(MEGA_DATA_DIR, exist_ok=True)

# 2. CHON DANH SACH CAC SHARDS CAN HUAN LUYEN
# - Phase 1 (50% Data): [1, 2, 4, 5, 6] (~112.000 anh)
# - Phase 2 (50% Data con lai): [0, 3, 7, 8, 9] (~112.000 anh)
# - Hoac toan bo 10 Shards: list(range(10)) (224.015 anh)
TARGET_SHARDS = [1, 2, 4, 5, 6]

print(f'\n--> [BUOC 1/4] Dang gom dong thoi {len(TARGET_SHARDS)} Shards: {TARGET_SHARDS}...')

for s in TARGET_SHARDS:
    tar_name = f'styleid-s-{s:06d}.tar'
    print(f'    + Dang tai {tar_name}...')
    tar_path = hf_hub_download(repo_id='kwanY/stylebench-s', filename=tar_name, repo_type='dataset', local_dir='./')
    
    last_png_bytes = None
    with tarfile.open(tar_path, 'r') as tar:
        for m in tqdm.tqdm(tar, desc=f'    Gop Shard {s}'):
            if m.name.endswith('.png'):
                f = tar.extractfile(m)
                if f is not None: last_png_bytes = f.read()
            elif m.name.endswith('.json') and last_png_bytes is not None:
                f = tar.extractfile(m)
                if f is not None:
                    try:
                        meta = json.loads(f.read().decode('utf-8'))
                        orig = meta.get('original_path')
                        if orig:
                            dest = os.path.join(MEGA_DATA_DIR, orig)
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            with open(dest, 'wb') as out_f: out_f.write(last_png_bytes)
                    except Exception: pass
                last_png_bytes = None
    try: os.remove(tar_path)
    except Exception: pass

total_train_imgs = len(glob.glob(f'{MEGA_DATA_DIR}/**/*.png', recursive=True))
print(f'\n--> DA GOP THANH CONG: {total_train_imgs:,} BUC ANH!')

# 3. DINH NGHIA KIEN TRUC DISENTANGLED-STYLEID
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
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()

# 4. DATASET & DATALOADER
class MultiStyleDataset(Dataset):
    def __init__(self, root):
        self.samples, self.class_to_idx, self.style_to_idx = [], {}, {}
        for p in glob.glob(os.path.join(root, '**', '*.png'), recursive=True):
            fname = os.path.basename(p)
            ident = fname.split('_')[0]
            cat = os.path.basename(os.path.dirname(p))
            if ident not in self.class_to_idx: self.class_to_idx[ident] = len(self.class_to_idx)
            if cat not in self.style_to_idx: self.style_to_idx[cat] = len(self.style_to_idx)
            self.samples.append((p, self.class_to_idx[ident], self.style_to_idx[cat]))
        print(f'--> Dataset nap xong: {len(self.samples):,} anh | {len(self.class_to_idx)} IDs | {len(self.style_to_idx)} Styles')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, id_l, st_l = self.samples[idx]
        return Image.open(p).convert('RGB'), id_l, st_l

def collate_fn(batch):
    return [b[0] for b in batch], torch.tensor([b[1] for b in batch], dtype=torch.long), torch.tensor([b[2] for b in batch], dtype=torch.long)

def get_features(m, pv): return m.visual_projection(m.vision_model(pixel_values=pv).pooler_output)

# 5. KHOI TAO VA NAP CHECKPOINT CU (NEU CO)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
for p in model.parameters(): p.requires_grad = False
apply_lora_to_clip(model, r=8)

train_dataset = MultiStyleDataset(MEGA_DATA_DIR)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)

num_styles = len(train_dataset.style_to_idx)
disentangle_head = DisentangledHead(num_styles=num_styles).to(device)
angular_head = AngularHead(512, len(train_dataset.class_to_idx)).to(device)

ckpt_prev_lora = f'{SAVE_DIR}/disentangled_lora_clip_a100_final.pt'
ckpt_prev_head = f'{SAVE_DIR}/disentangled_head_a100_final.pt'
if os.path.exists(ckpt_prev_lora):
    print(f'--> Phat hien checkpoint cu, nap tiep noi: {ckpt_prev_lora}')
    model.load_state_dict(torch.load(ckpt_prev_lora, map_location=device), strict=False)
    old_h = torch.load(ckpt_prev_head, map_location=device)
    cur_h = disentangle_head.state_dict()
    for k, v in old_h.items():
        if k in cur_h and cur_h[k].shape == v.shape: cur_h[k] = v.to(device)
    disentangle_head.load_state_dict(cur_h)

trainable_params = [p for p in model.parameters() if p.requires_grad] + list(disentangle_head.parameters()) + list(angular_head.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-4, weight_decay=1e-4)
epochs = 2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda')
ce_loss, supcon_fn = nn.CrossEntropyLoss(), SupConLoss()

# 6. VONG LAP HUAN LUYEN
print(f'\n--> [BUOC 3/4] BAT DAU HUAN LUYEN TREN {total_train_imgs:,} ANH...')
for epoch in range(epochs):
    model.train(); disentangle_head.train(); angular_head.train()
    pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (A100)')
    total_loss, steps = 0.0, 0
    for images, id_labels, style_labels in pbar:
        id_labels, style_labels = id_labels.to(device), style_labels.to(device)
        pv = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feat = get_features(model, pv)
            z_id, z_style, style_logits = disentangle_head(feat)
            loss_id = ce_loss(angular_head(z_id, id_labels), id_labels) + 0.5 * supcon_fn(z_id, id_labels)
            loss_style = ce_loss(style_logits, style_labels)
            loss_ortho = disentangle_head.compute_ortho_loss(z_id, z_style)
            loss = loss_id + 0.5 * loss_style + 0.1 * loss_ortho
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item(); steps += 1
        pbar.set_postfix({'loss': f'{total_loss/steps:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

# 7. LUU CHECKPOINT VAO GOOGLE DRIVE
lora_save_path = f'{SAVE_DIR}/disentangled_lora_clip_a100_final.pt'
head_save_path = f'{SAVE_DIR}/disentangled_head_a100_final.pt'
torch.save(model.state_dict(), lora_save_path)
torch.save(disentangle_head.state_dict(), head_save_path)
print(f'--> DA LUU THANH CONG: {lora_save_path} V\u00c0 {head_save_path}!')
