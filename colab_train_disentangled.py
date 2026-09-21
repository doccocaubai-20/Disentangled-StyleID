import os, glob, json, tarfile, shutil, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
from huggingface_hub import hf_hub_download
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'--> Đang sử dụng thiết bị: {device.upper()}')

TEMP_TRAIN_DIR = "temp_train_data"
VAL_DIR = "permanent_val_data"
DRIVE_SAVE_DIR = "/content/drive/MyDrive"
#1
def extract_shard(shard_idx):
    tar_filename = f"styleid-s-{shard_idx:06d}.tar"
    print(f"\n{'='*65}")
    print(f" [BƯỚC 1] ĐANG TẢI SHARD {shard_idx}: {tar_filename}...")
    print(f"{'='*65}")

    tar_path = hf_hub_download(
        repo_id="kwanY/stylebench-s",
        filename=tar_filename,
        repo_type="dataset",
        local_dir="./"
    )

    print(f"--> Đang giải nén Shard {shard_idx} vào thư mục tạm...")
    os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    val_ready = len(glob.glob(f"{VAL_DIR}/**/*.png", recursive=True)) >= 500
    total_imgs, last_png_bytes = 0, None

    with tarfile.open(tar_path, 'r') as tar:
        for m in tqdm.tqdm(tar, desc=f"Bung ảnh Shard {shard_idx}"):
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
                            if not val_ready and total_imgs < 500:
                                dest = os.path.join(VAL_DIR, orig)
                            else:
                                dest = os.path.join(TEMP_TRAIN_DIR, orig)

                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            with open(dest, "wb") as out_f: out_f.write(last_png_bytes)
                            total_imgs += 1
                    except Exception: pass
                last_png_bytes = None

    try:
        os.remove(tar_path)
        print(f"--> XOA .TAR!")
    except Exception: pass

    train_count = len(glob.glob(f"{TEMP_TRAIN_DIR}/**/*.png", recursive=True))
    print(f"--> Shard {shard_idx} sẵn sàng: {train_count} ảnh huấn luyện.")

def cleanup_temp_train():
    if os.path.exists(TEMP_TRAIN_DIR):
        shutil.rmtree(TEMP_TRAIN_DIR)
        print(f"XOA TEMP SHARD")

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
#2
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
        return F.normalize(self.id_proj(feat), dim=-1), F.normalize(self.style_proj(feat), dim=-1), self.style_classifier(F.normalize(self.style_proj(feat), dim=-1))
    def compute_ortho_loss(self, z_id, z_style):
        z_id_proj = F.normalize(self.ortho_proj(z_id), dim=-1)
        return ((z_id_proj * z_style).sum(dim=-1) ** 2).mean()
# 3
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
        features = F.normalize(features, dim=1)
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

def evaluate_val_set(model, disentangle_head, processor):
    val_dataset = MultiStyleDataset(VAL_DIR)
    val_embs, id_lbls, sty_lbls = [], [], []
    model.eval(); disentangle_head.eval()
    with torch.no_grad():
        for i in range(min(len(val_dataset), 400)):
            img, id_lbl, sty_lbl = val_dataset[i]
            pv = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            z_id, _, _ = disentangle_head(get_features(model, pv))
            val_embs.append(z_id); id_lbls.append(id_lbl); sty_lbls.append(sty_lbl)

    val_embs = torch.cat(val_embs, dim=0)
    sim_matrix = (val_embs @ val_embs.T).cpu()
    pos_sims, neg_sims = [], []
    for i in range(len(id_lbls)):
        for j in range(i + 1, len(id_lbls)):
            s = sim_matrix[i, j].item()
            if id_lbls[i] == id_lbls[j]: pos_sims.append(s)
            else: neg_sims.append(s)
    random.seed(42)
    neg_sampled = random.sample(neg_sims, min(len(neg_sims), len(pos_sims) * 3))
    acc_04 = (sum([1 for s in pos_sims if s >= 0.4]) + sum([1 for s in neg_sampled if s < 0.4])) / (len(pos_sims) + len(neg_sampled))
    return acc_04, np.mean(pos_sims), np.mean(neg_sampled)

def train_shard(shard_idx=3, epochs_per_shard=2, batch_size=28, accumulation_steps=4):
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    for p in model.parameters(): p.requires_grad = False
    apply_lora_to_clip(model, r=8)

    # 1. Nạp lại checkpoint
    if os.path.exists('disentangled_lora_clip.pt'):
        print('--> Đang nạp tiếp checkpoint LoRA (1.6 GB)...')
        model.load_state_dict(torch.load('disentangled_lora_clip.pt', map_location=device), strict=False)

    disentangle_head = DisentangledHead(clip_dim=model.config.projection_dim, id_dim=512, style_dim=256).to(device)
    if os.path.exists('disentangled_head.pt'):
        print('--> Đang nạp tiếp checkpoint DisentangledHead...')
        old_state = torch.load('disentangled_head.pt', map_location=device)
        cur_state = disentangle_head.state_dict()
        for k, v in old_state.items():
            if k in cur_state and cur_state[k].shape == v.shape: cur_state[k] = v
        disentangle_head.load_state_dict(cur_state)

    ce_loss, supcon_fn = nn.CrossEntropyLoss(), SupConLoss().to(device)

    print(f"\n{'#'*65}")
    print(f"   BẮT ĐẦU HUẤN LUYỆN SHARD {shard_idx}")
    print(f"{'#'*65}")

    # 2. Tải và bung shard
    extract_shard(shard_idx)

    # 3. Huấn luyện
    train_dataset = MultiStyleDataset(TEMP_TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    angular_head = AngularHead(512, len(train_dataset.class_to_idx)).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(disentangle_head.parameters()) + list(angular_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-4)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs_per_shard):
        model.train(); disentangle_head.train()
        optimizer.zero_grad()
        pbar = tqdm.tqdm(train_loader, desc=f"Shard {shard_idx} | Epoch {epoch+1}/{epochs_per_shard}")
        for step, (images, id_labels, style_labels) in enumerate(pbar):
            id_labels, style_labels = id_labels.to(device), style_labels.to(device)
            pv = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
            with torch.amp.autocast('cuda'):
                feat = get_features(model, pv)
                z_id, z_style, style_logits = disentangle_head(feat)
                loss_id = ce_loss(angular_head(z_id, id_labels), id_labels) + supcon_fn(z_id, id_labels)
                loss_style = ce_loss(style_logits, style_labels)
                loss_ortho = disentangle_head.compute_ortho_loss(z_id, z_style)
                loss = (loss_id + 0.5 * loss_style + 0.1 * loss_ortho) / accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            pbar.set_postfix({'loss': f"{(loss.item()*accumulation_steps):.4f}"})

    # 4. Lưu cục bộ và TỰ ĐỘNG ĐẨY THẲNG VỀ GOOGLE DRIVE
    torch.save(model.state_dict(), 'disentangled_lora_clip.pt')
    torch.save(disentangle_head.state_dict(), 'disentangled_head.pt')


    # 5. Dọn dẹp ổ cứng
    cleanup_temp_train()

    # 6. Đánh giá kiểm thử
    acc, pos_s, neg_s = evaluate_val_set(model, disentangle_head, processor)
    print(f"\n>>> [TIẾN ĐỘ SAU SHARD {shard_idx}]: Acc: {acc*100:.2f}% | Cùng người: {pos_s:.4f} | Khác người: {neg_s:.4f} (Margin: {pos_s-neg_s:.4f}) <<<\n")

if __name__ == '__main__':
    train_shard(shard_idx=7, epochs_per_shard=2)