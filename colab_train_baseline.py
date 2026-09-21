import os
import glob
import json
import tarfile
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
from huggingface_hub import hf_hub_download
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Đang sử dụng thiết bị: {device.upper()}")

DATA_ROOT = "styleid-s"

# ============================================================
# 1. Tự động trích xuất tập dữ liệu Đa phong cách (Multi-Style)
# ============================================================
def prepare_multistyle_data(max_per_style=1200):
    tar_filename = "styleid-s-000000.tar"
    if not os.path.exists(tar_filename):
        print("--> Đang tải shard styleid-s-000000.tar (~7GB) từ HuggingFace...")
        tar_path = hf_hub_download(
            repo_id="kwanY/stylebench-s",
            filename=tar_filename,
            repo_type="dataset",
            local_dir="./"
        )
    else:
        tar_path = tar_filename

    # Kiểm tra xem đã có đủ ít nhất 2 styles trong thư mục chưa
    styles_found = set()
    for root_dir, dirs, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith(".png"):
                parts = root_dir.replace("\\", "/").split("/")
                if len(parts) >= 3:
                    styles_found.add(parts[2])

    if len(styles_found) >= 2:
        total_pngs = len(glob.glob(f"{DATA_ROOT}/**/*.png", recursive=True))
        print(f"--> Đã tìm thấy {len(styles_found)} styles sẵn có: {styles_found} ({total_pngs} ảnh). Bỏ qua bước giải nén.")
        return

    print("--> Đang giải nén tập ĐA PHONG CÁCH (Pixar, Anime, Caricature)...")
    os.makedirs(DATA_ROOT, exist_ok=True)

    style_counts = {}
    last_png_bytes = None

    with tarfile.open(tar_path, "r") as tar:
        for m in tqdm.tqdm(tar, desc="Đang trích xuất đa phong cách"):
            if m.name.endswith(".png"):
                f = tar.extractfile(m)
                if f is not None:
                    last_png_bytes = f.read()
            elif m.name.endswith(".json") and last_png_bytes is not None:
                f = tar.extractfile(m)
                if f is not None:
                    try:
                        meta = json.loads(f.read().decode("utf-8"))
                        orig_path = meta.get("original_path")
                        if orig_path:
                            parts = orig_path.replace("\\", "/").split("/")
                            cat = parts[1] if len(parts) >= 2 else "Unknown"

                            # Lấy tối đa max_per_style ảnh cho mỗi phong cách để cân bằng
                            if style_counts.get(cat, 0) < max_per_style:
                                dest_path = os.path.join(DATA_ROOT, orig_path)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                with open(dest_path, "wb") as out_f:
                                    out_f.write(last_png_bytes)
                                style_counts[cat] = style_counts.get(cat, 0) + 1
                    except Exception:
                        pass
                last_png_bytes = None

    print(f"--> Hoàn tất! Số ảnh theo từng phong cách: {style_counts}")

# ============================================================
# 2. Dataset đọc Identity & Style Label
# ============================================================
class MultiStyleDataset(Dataset):
    def __init__(self, root, is_train=True, val_split=0.2, seed=42):
        self.samples = []
        self.class_to_idx = {}
        self.style_to_idx = {}
        all_samples = []

        style_methods = sorted(os.listdir(root))
        for method in style_methods:
            method_path = os.path.join(root, method)
            if not os.path.isdir(method_path):
                continue
            for cat in sorted(os.listdir(method_path)):
                cat_path = os.path.join(method_path, cat)
                if not os.path.isdir(cat_path):
                    continue
                for p in glob.glob(os.path.join(cat_path, "*.png")):
                    fname = os.path.basename(p)
                    ident = fname.split("_")[0]
                    all_samples.append((p, ident, cat))

        # Phân chia Identity Unseen
        unique_ids = sorted(list(set([s[1] for s in all_samples])))
        random.seed(seed)
        random.shuffle(unique_ids)

        split_idx = int(len(unique_ids) * (1 - val_split))
        train_ids = set(unique_ids[:split_idx])
        val_ids = set(unique_ids[split_idx:])

        unique_styles = sorted(list(set([s[2] for s in all_samples])))
        for idx, sty in enumerate(unique_styles):
            self.style_to_idx[sty] = idx

        target_ids = train_ids if is_train else val_ids
        for path, ident, style in all_samples:
            if ident in target_ids:
                if ident not in self.class_to_idx:
                    self.class_to_idx[ident] = len(self.class_to_idx)
                self.samples.append((path, self.class_to_idx[ident], self.style_to_idx[style], ident, style))

        split_name = "TRAIN" if is_train else "VAL (UNSEEN)"
        print(f"[{split_name}] {len(self.samples)} ảnh | {len(self.class_to_idx)} IDs | Styles: {list(self.style_to_idx.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, id_lbl, sty_lbl, ident, style = self.samples[idx]
        img = load_image(path).convert("RGB")
        return img, id_lbl, sty_lbl

def collate_fn(batch):
    images, id_labels, style_labels = zip(*batch)
    return list(images), torch.tensor(id_labels, dtype=torch.long)

# ============================================================
# 3. LoRA Module
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.scaling = alpha / r
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

class AngularHead(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=0.3, scale=30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.margin = margin
        self.scale = scale

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=-1)
        W_norm = F.normalize(self.W, dim=-1)
        logits = x_norm @ W_norm.t()
        if labels is None:
            return logits
        theta = torch.acos(logits.clamp(-1 + 1e-5, 1 - 1e-5))
        target_logits = torch.cos(theta + self.margin)
        onehot = torch.zeros_like(logits)
        onehot.scatter_(1, labels.unsqueeze(1), 1)
        logits = logits * (1 - onehot) + target_logits * onehot
        return logits * self.scale

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        return -((mask * log_prob).sum(1) / mask_pos_pairs).mean()

# ============================================================
# 4. Đánh giá chuyên sâu: Phân tách rõ In-Style vs Cross-Style
# ============================================================
def evaluate_cross_style(val_embs, val_id_labels, val_style_labels):
    sim_matrix = (val_embs @ val_embs.T).cpu()
    
    in_style_pos = []     # Cùng người, CÙNG phong cách
    cross_style_pos = []  # Cùng người, KHÁC phong cách (QUAN TRỌNG NHẤT!)
    neg_sims = []         # Khác người

    for i in range(len(val_id_labels)):
        for j in range(i + 1, len(val_id_labels)):
            s = sim_matrix[i, j].item()
            if val_id_labels[i] == val_id_labels[j]:
                if val_style_labels[i] == val_style_labels[j]:
                    in_style_pos.append(s)
                else:
                    cross_style_pos.append(s)
            else:
                neg_sims.append(s)

    random.seed(42)
    all_pos = in_style_pos + cross_style_pos
    neg_sampled = random.sample(neg_sims, min(len(neg_sims), len(all_pos) * 3))

    # Tính độ chính xác Cross-Style (Người giống nhau nhưng ở 2 style khác nhau)
    if cross_style_pos:
        cross_acc_04 = sum([1 for s in cross_style_pos if s >= 0.4]) / len(cross_style_pos)
        mean_cross_pos = np.mean(cross_style_pos)
    else:
        cross_acc_04, mean_cross_pos = 0.0, 0.0

    mean_neg = np.mean(neg_sampled) if neg_sampled else 0.0
    overall_acc_04 = (sum([1 for s in all_pos if s >= 0.4]) + sum([1 for s in neg_sampled if s < 0.4])) / (len(all_pos) + len(neg_sampled))

    return mean_cross_pos, mean_neg, cross_acc_04, overall_acc_04, len(cross_style_pos), len(in_style_pos)

def get_features(m, inp):
    if hasattr(inp, 'pixel_values'):
        pixel_values = inp.pixel_values
    elif isinstance(inp, dict) and 'pixel_values' in inp:
        pixel_values = inp['pixel_values']
    elif hasattr(inp, '__getitem__') and not isinstance(inp, torch.Tensor):
        pixel_values = inp['pixel_values']
    else:
        pixel_values = inp
    vision_outputs = m.vision_model(pixel_values=pixel_values)
    return m.visual_projection(vision_outputs.pooler_output)

# ============================================================
# 5. Huấn luyện Baseline
# ============================================================
def train_and_eval_baseline(epochs=5, batch_size=32, lr=2e-4):
    prepare_multistyle_data()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora_to_clip(model, r=8)

    train_dataset = MultiStyleDataset(DATA_ROOT, is_train=True)
    val_dataset = MultiStyleDataset(DATA_ROOT, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    angular_head = AngularHead(model.config.projection_dim, len(train_dataset.class_to_idx)).to(device)
    supcon_fn = SupConLoss().to(device)
    ce_loss = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(angular_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    print(f"\n===== HUẤN LUYỆN BASELINE STYLEID GỐC TRÊN ĐA PHONG CÁCH ({epochs} EPOCHS) =====")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            labels = labels.to(device)
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    base_emb = F.normalize(get_features(model, pixel_values), dim=-1)
                new_emb = get_features(model, pixel_values)
                logits = angular_head(new_emb, labels)
                loss_ang = ce_loss(logits, labels)
                loss_contra = supcon_fn(new_emb, labels)
                new_norm = F.normalize(new_emb, dim=-1)
                loss_align = 1.0 - (new_norm * base_emb).sum(dim=-1).mean()
                loss = 1.0 * loss_ang + 1.0 * loss_contra + 0.1 * loss_align

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Lưu trọng số Baseline
    torch.save(model.state_dict(), "baseline_lora_clip.pt")
    print("--> Đã lưu checkpoint: baseline_lora_clip.pt")

    # Đánh giá Cross-Style
    print("\n--> Đang đánh giá kiểm thử Cross-Style...")
    model.eval()
    val_embs, val_id_labels, val_style_labels = [], [], []
    with torch.no_grad():
        for i in range(min(len(val_dataset), 500)):
            img, id_lbl, sty_lbl = val_dataset[i]
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            emb = get_features(model, pixel_values)
            emb = F.normalize(emb, dim=-1)
            val_embs.append(emb)
            val_id_labels.append(id_lbl)
            val_style_labels.append(sty_lbl)

    val_embs = torch.cat(val_embs, dim=0)
    mean_cross_pos, mean_neg, cross_acc, overall_acc, n_cross, n_in = evaluate_cross_style(
        val_embs, val_id_labels, val_style_labels
    )

    print(f"\n=======================================================")
    print(f" [KẾT QUẢ BASELINE (STYLEID GỐC) - ĐA PHONG CÁCH]")
    print(f" - ĐỘ CHÍNH XÁC CROSS-STYLE (Khác phong cách):    {cross_acc * 100:.2f}%")
    print(f" - Độ chính xác tổng thể (Overall Accuracy@0.4):   {overall_acc * 100:.2f}%")
    print(f" - Độ tương đồng CÙNG NGƯỜI (Khác style):         {mean_cross_pos:.4f}")
    print(f" - Độ tương đồng KHÁC NGƯỜI:                       {mean_neg:.4f}")
    print(f" - Số cặp test: {n_cross} cặp Cross-Style / {n_in} cặp cùng style")
    print(f"=======================================================\n")

if __name__ == "__main__":
    train_and_eval_baseline()
