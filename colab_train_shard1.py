# Huấn luyện tiếp tục Shard 1 (Sequential Shard Training)
import os, glob, json, tarfile, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
from huggingface_hub import hf_hub_download
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'--> Đang sử dụng thiết bị: {device.upper()}')
DATA_ROOT = 'styleid-s'

# 1. Tự động dọn dẹp và tải Shard 1
def prepare_shard1_data(max_per_style=1200):
    # Xóa file nén shard 0 cũ nếu còn để giải phóng 7GB ổ cứng
    if os.path.exists('styleid-s-000000.tar'):
        try:
            os.remove('styleid-s-000000.tar')
            print('--> Đã xóa styleid-s-000000.tar để giải phóng ổ cứng.')
        except Exception: pass

    tar_filename = 'styleid-s-000001.tar'
    if not os.path.exists(tar_filename):
        print('--> Đang tải shard styleid-s-000001.tar (~6.6GB) từ HuggingFace...')
        tar_path = hf_hub_download(
            repo_id='kwanY/stylebench-s',
            filename=tar_filename,
            repo_type='dataset',
            local_dir='./'
        )
    else:
        tar_path = tar_filename

    print('--> Đang trích xuất dữ liệu từ Shard 1 vào styleid-s...')
    os.makedirs(DATA_ROOT, exist_ok=True)
    style_counts, last_png_bytes = {}, None

    with tarfile.open(tar_path, 'r') as tar:
        for m in tqdm.tqdm(tar, desc='Trích xuất Shard 1'):
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
                            parts = orig.replace('\\', '/').split('/')
                            cat = parts[1] if len(parts) >= 2 else 'Unknown'
                            if style_counts.get(cat, 0) < max_per_style:
                                dest = os.path.join(DATA_ROOT, orig)
                                os.makedirs(os.path.dirname(dest), exist_ok=True)
                                with open(dest, 'wb') as out_f: out_f.write(last_png_bytes)
                                style_counts[cat] = style_counts.get(cat, 0) + 1
                    except Exception: pass
                last_png_bytes = None

    print(f'--> Hoàn tất trích xuất Shard 1: {style_counts}')
    # Xóa file tar vừa tải để ổ cứng luôn trống > 45GB
    try:
        os.remove(tar_path)
        print('--> Đã tự động xóa file .tar của Shard 1 để giữ ổ cứng an toàn!')
    except Exception: pass

# 2. Dataset đa phong cách
class MultiStyleDataset(Dataset):
    def __init__(self, root, is_train=True, val_split=0.2, seed=42):
        self.samples = []
        self.class_to_idx = {}
        self.style_to_idx = {}
        all_samples = []

        style_methods = sorted(os.listdir(root))
        for method in style_methods:
            method_path = os.path.join(root, method)
            if not os.path.isdir(method_path): continue
            for cat in sorted(os.listdir(method_path)):
                cat_path = os.path.join(method_path, cat)
                if not os.path.isdir(cat_path): continue
                for p in glob.glob(os.path.join(cat_path, '*.png')):
                    fname = os.path.basename(p)
                    ident = fname.split('_')[0]
                    all_samples.append((p, ident, cat))

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

        split_name = 'TRAIN' if is_train else 'VAL (UNSEEN)'
        print(f'[{split_name}] {len(self.samples)} ảnh | {len(self.class_to_idx)} IDs | Styles ({len(self.style_to_idx)}): {list(self.style_to_idx.keys())}')

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, id_lbl, sty_lbl, ident, style = self.samples[idx]
        img = load_image(path).convert('RGB')
        return img, id_lbl, sty_lbl

def collate_fn(batch):
    images, id_labels, style_labels = zip(*batch)
    return list(images), torch.tensor(id_labels, dtype=torch.long), torch.tensor(style_labels, dtype=torch.long)

# 3. LoRA Module
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

class DisentangledHead(nn.Module):
    def __init__(self, clip_dim=768, id_dim=512, style_dim=256, num_styles=3):
        super().__init__()
        self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
        self.style_proj = nn.Sequential(nn.Linear(clip_dim, style_dim), nn.BatchNorm1d(style_dim), nn.PReLU())
        self.style_classifier = nn.Linear(style_dim, max(num_styles, 2))
        self.ortho_proj = nn.Linear(id_dim, style_dim, bias=False)
    def forward(self, feat):
        z_id = F.normalize(self.id_proj(feat), dim=-1)
        z_style = F.normalize(self.style_proj(feat), dim=-1)
        style_logits = self.style_classifier(z_style)
        return z_id, z_style, style_logits
    def compute_ortho_loss(self, z_id, z_style):
        z_id_proj = F.normalize(self.ortho_proj(z_id), dim=-1)
        dot_product = (z_id_proj * z_style).sum(dim=-1)
        return (dot_product ** 2).mean()

class AngularHead(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=0.3, scale=30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.margin, self.scale = margin, scale
    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=-1)
        W_norm = F.normalize(self.W, dim=-1)
        logits = x_norm @ W_norm.t()
        if labels is None: return logits
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

def get_features(m, pixel_values):
    vision_outputs = m.vision_model(pixel_values=pixel_values)
    return m.visual_projection(vision_outputs.pooler_output)

def evaluate_cross_style(val_embs, val_id_labels, val_style_labels):
    sim_matrix = (val_embs @ val_embs.T).cpu()
    in_style_pos, cross_style_pos, neg_sims = [], [], []
    for i in range(len(val_id_labels)):
        for j in range(i + 1, len(val_id_labels)):
            s = sim_matrix[i, j].item()
            if val_id_labels[i] == val_id_labels[j]:
                if val_style_labels[i] == val_style_labels[j]: in_style_pos.append(s)
                else: cross_style_pos.append(s)
            else: neg_sims.append(s)
    random.seed(42)
    all_pos = in_style_pos + cross_style_pos
    neg_sampled = random.sample(neg_sims, min(len(neg_sims), len(all_pos) * 3))
    cross_acc = sum([1 for s in cross_style_pos if s >= 0.4]) / len(cross_style_pos) if cross_style_pos else 0.0
    mean_cross = np.mean(cross_style_pos) if cross_style_pos else 0.0
    mean_neg = np.mean(neg_sampled) if neg_sampled else 0.0
    overall_acc = (sum([1 for s in all_pos if s >= 0.4]) + sum([1 for s in neg_sampled if s < 0.4])) / (len(all_pos) + len(neg_sampled))
    return mean_cross, mean_neg, cross_acc, overall_acc, len(cross_style_pos), len(in_style_pos)

# 4. Vòng lặp huấn luyện nối tiếp Shard 1
def train_and_eval_shard1(epochs=5, batch_size=28, accumulation_steps=4, lr=1.5e-4):
    prepare_shard1_data()

    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    for p in model.parameters(): p.requires_grad = False
    apply_lora_to_clip(model, r=8)

    # Nạp tiếp trọng số LoRA từ Shard 0 nếu có
    if os.path.exists('disentangled_lora_clip.pt'):
        print('--> Nạp tiếp trọng số LoRA từ Shard 0 (disentangled_lora_clip.pt)...')
        model.load_state_dict(torch.load('disentangled_lora_clip.pt', map_location=device), strict=False)

    train_dataset = MultiStyleDataset(DATA_ROOT, is_train=True)
    val_dataset = MultiStyleDataset(DATA_ROOT, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    num_ids = len(train_dataset.class_to_idx)
    num_styles = len(train_dataset.style_to_idx)

    disentangle_head = DisentangledHead(
        clip_dim=model.config.projection_dim,
        id_dim=512,
        style_dim=256,
        num_styles=num_styles
    ).to(device)

    # Nạp tiếp trọng số DisentangledHead từ Shard 0
    if os.path.exists('disentangled_head.pt'):
        print('--> Nạp tiếp các tầng chiếu Disentangled từ Shard 0 (disentangled_head.pt)...')
        old_state = torch.load('disentangled_head.pt', map_location=device)
        cur_state = disentangle_head.state_dict()
        for k, v in old_state.items():
            if k in cur_state and cur_state[k].shape == v.shape:
                cur_state[k] = v
        disentangle_head.load_state_dict(cur_state)

    id_angular_head = AngularHead(512, num_ids).to(device)
    supcon_fn = SupConLoss().to(device)
    ce_loss = nn.CrossEntropyLoss()

    trainable_params = (
        [p for p in model.parameters() if p.requires_grad] +
        list(disentangle_head.parameters()) +
        list(id_angular_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    print(f'\n===== HUẤN LUYỆN NỐI TIẾP SHARD 1 (GRADIENT ACCUMULATION BATCH 112) - {epochs} EPOCHS =====')
    for epoch in range(epochs):
        model.train()
        disentangle_head.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for step, (images, id_labels, style_labels) in enumerate(pbar):
            id_labels = id_labels.to(device)
            style_labels = style_labels.to(device)
            inputs = processor(images=images, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(device)

            with torch.amp.autocast('cuda'):
                feat = get_features(model, pixel_values)
                z_id, z_style, style_logits = disentangle_head(feat)

                loss_id_ang = ce_loss(id_angular_head(z_id, id_labels), id_labels)
                loss_id_scon = supcon_fn(z_id, id_labels)
                loss_style_ce = ce_loss(style_logits, style_labels)
                loss_ortho = disentangle_head.compute_ortho_loss(z_id, z_style)

                loss = (1.0 * loss_id_ang + 1.0 * loss_id_scon + 0.5 * loss_style_ce + 0.1 * loss_ortho) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'id': f'{(loss_id_ang+loss_id_scon).item():.3f}', 'sty': f'{loss_style_ce.item():.3f}', 'ortho': f'{loss_ortho.item():.4f}'})

    # Lưu lại checkpoint mới sau khi học thêm Shard 1
    torch.save(model.state_dict(), 'disentangled_lora_clip.pt')
    torch.save(disentangle_head.state_dict(), 'disentangled_head.pt')
    print('\n--> ĐÃ LƯU CHECKPOINT NÂNG CẤP: disentangled_lora_clip.pt & disentangled_head.pt')

    # Đánh giá kiểm thử
    print('\n--> Đang đánh giá kiểm thử Cross-Style trên tập mở rộng...')
    model.eval()
    disentangle_head.eval()
    val_embs, val_id_labels, val_style_labels = [], [], []

    with torch.no_grad():
        for i in range(min(len(val_dataset), 600)):
            img, id_lbl, sty_lbl = val_dataset[i]
            inputs = processor(images=img, return_tensors='pt')
            pv = inputs['pixel_values'].to(device)
            feat = get_features(model, pv)
            z_id, _, _ = disentangle_head(feat)
            val_embs.append(z_id)
            val_id_labels.append(id_lbl)
            val_style_labels.append(sty_lbl)

    val_embs = torch.cat(val_embs, dim=0)
    mean_cross_pos, mean_neg, cross_acc, overall_acc, n_cross, n_in = evaluate_cross_style(
        val_embs, val_id_labels, val_style_labels
    )

    print('\n' + '='*65)
    print(' [KẾT QUẢ DISENTANGLED-STYLEID SAU KHI HỌC SHARD 1]')
    print(f' - ĐỘ CHÍNH XÁC CROSS-STYLE (Khác phong cách):    {cross_acc * 100:.2f}%')
    print(f' - Độ chính xác tổng thể (Overall Accuracy@0.4):   {overall_acc * 100:.2f}%')
    print(f' - Độ tương đồng CÙNG NGƯỜI (Khác style):         {mean_cross_pos:.4f}')
    print(f' - Độ tương đồng KHÁC NGƯỜI:                       {mean_neg:.4f}')
    print(f' - Số cặp test: {n_cross} cặp Cross-Style / {n_in} cặp cùng style')
    print('='*65 + '\n')

if __name__ == '__main__':
    train_and_eval_shard1()
