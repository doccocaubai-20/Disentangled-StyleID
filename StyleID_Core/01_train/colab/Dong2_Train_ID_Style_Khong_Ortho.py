"""Pooled A100-style five-shard training without orthogonality loss.
ID512 and style256 supervision retained. No checkpoint mixing, no Soup init.
This is a new experiment; it does not reproduce the historical Soup model.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from google.colab import drive

if not Path('/content/drive/MyDrive').is_dir(): drive.mount('/content/drive')
SEED = int(globals().get('SEED', 20261024))
BATCH_SIZE = int(globals().get('BATCH_SIZE', 128))
EPOCHS = int(globals().get('EPOCHS', 2))
if BATCH_SIZE < 2 or EPOCHS < 1: raise ValueError('Invalid batch size or epochs')
BASE_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
RUN_NAME = 'id_style_no_ortho_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
RUN_DIR = Path(globals().get('OUTPUT_ROOT','/content/drive/MyDrive/StyleID_NoOrtho'))/RUN_NAME
RUN_DIR.mkdir(parents=True,exist_ok=False)
WORK_DIR = Path('/content/styleid_no_ortho_data')/RUN_NAME
WORK_DIR.mkdir(parents=True,exist_ok=False)
recipe = dict(seed=SEED,shards=[1,2,4,5,6],batch_size=BATCH_SIZE,epochs=EPOCHS,
    data_order='All five shards pooled and globally shuffled each epoch',
    initialization='OpenAI CLIP; fresh head and LoRA',base_revision=BASE_REVISION,
    loss='AngularCE + SupCon + 0.5*StyleCE',ortho_weight=0.0,
    optimizer='AdamW',learning_rate=1.2e-4,weight_decay=1e-4,
    scheduler='CosineAnnealingLR',eta_min=1e-6,precision='BF16',
    lora_rank=8,lora_alpha=1.0,angular_margin=0.3,angular_scale=30,temperature=0.1,
    checkpoint_selection='Final epoch; no benchmark selection',
    resume='Not supported; rerunning starts a new run',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    versions={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy')})

import os, glob, json, tarfile, shutil, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image
import tqdm

# GPU training only; initialize a fresh model, no historical weights.
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('Select a CUDA GPU with BF16 support (recommended: A100).')
device = 'cuda'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
SAVE_DIR = str(RUN_DIR)
MEGA_DATA_DIR = str(WORK_DIR/'mega_train_data')
os.makedirs(MEGA_DATA_DIR, exist_ok=False)
print('Output:', RUN_DIR, flush=True)

# 2. Tải và gộp đồng thời các Shard lớn nhất vào chung 1 thư mục
TARGET_SHARDS = [1, 2, 4, 5, 6]  # Five of ten shards; image fraction is not necessarily 50%.
print(f"\n--> [BƯỚC 1/4] Đang gom đồng thời {len(TARGET_SHARDS)} Shards lớn vào chung 1 thư mục...")

for s in TARGET_SHARDS:
    tar_name = f"styleid-s-{s:06d}.tar"
    print(f"    + Đang kéo {tar_name}...")
    tar_path = hf_hub_download(repo_id="kwanY/stylebench-s", filename=tar_name, repo_type="dataset", local_dir=str(WORK_DIR/"archives"))
    
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
                            dest = str((Path(MEGA_DATA_DIR)/orig).resolve())
                            if not Path(dest).is_relative_to(Path(MEGA_DATA_DIR).resolve()):
                                raise ValueError('Archive original_path escapes data directory')
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            with open(dest, "wb") as out_f: out_f.write(last_png_bytes)
                    except Exception as exc: raise RuntimeError(f"Invalid sample in {tar_name}: {m.name}") from exc
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
                for p in sorted(glob.glob(os.path.join(cp, "*.png"))):
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
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', revision=BASE_REVISION)
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', revision=BASE_REVISION).to(device)
for p in model.parameters(): p.requires_grad = False
apply_lora_to_clip(model, r=8)

head = DisentangledHead(clip_dim=model.config.projection_dim, id_dim=512, style_dim=256).to(device)
for p in head.ortho_proj.parameters(): p.requires_grad = False
ce_loss, supcon_fn = nn.CrossEntropyLoss(), SupConLoss().to(device)

train_dataset = MultiStyleDataset(MEGA_DATA_DIR)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,  # BATCH SIZE 128 CỰC MẠNH
    collate_fn=collate_fn, drop_last=True, num_workers=0, pin_memory=True
)
if len(train_dataset) < BATCH_SIZE:
    raise ValueError('Not enough images for one full training batch')
recipe.update(image_count=len(train_dataset),identity_count=len(train_dataset.class_to_idx),
              style_count=len(train_dataset.style_to_idx),
              class_to_idx=train_dataset.class_to_idx,style_to_idx=train_dataset.style_to_idx)
(RUN_DIR/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8')
angular_head = AngularHead(512, len(train_dataset.class_to_idx)).to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad] + list(angular_head.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1.2e-4, weight_decay=1e-4)
epochs = EPOCHS
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-6)

print(f"Train {epochs} epochs on {len(train_dataset)} pooled images; no orthogonality loss", flush=True)
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
            loss = loss_id + 0.5 * loss_style

        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

# Save a direct-trained model. Do not overwrite historical Soup checkpoints.
torch.save(model.state_dict(), RUN_DIR/'disentangled_lora_clip_no_ortho.pt')
torch.save(head.state_dict(), RUN_DIR/'disentangled_head_no_ortho.pt')
completed = dict(recipe, status='complete', checkpoint_sha256={
    name:hashlib.sha256((RUN_DIR/name).read_bytes()).hexdigest()
    for name in ('disentangled_lora_clip_no_ortho.pt','disentangled_head_no_ortho.pt')})
(RUN_DIR/'completed.json').write_text(json.dumps(completed,ensure_ascii=False,indent=2),encoding='utf-8')
# Remove only this run's temporary extracted images/archives after successful save.
if WORK_DIR.resolve().parent != Path('/content/styleid_no_ortho_data').resolve():
    raise RuntimeError('Unexpected cleanup directory')
shutil.rmtree(WORK_DIR)
print('COMPLETE. Direct model saved on Drive; temporary Colab images removed:', RUN_DIR, flush=True)
