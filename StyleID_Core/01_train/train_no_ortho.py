"""Train the pooled ID512 + style256 control with orthogonality loss disabled."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import tarfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
from tqdm import tqdm


BASE_MODEL = 'openai/clip-vit-large-patch14'
BASE_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
TARGET_SHARDS = (1, 2, 4, 5, 6)
CORE_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=CORE_DIR / 'checkpoints' / 'no_ortho',
                        help='Parent directory for the new run and its checkpoints')
    parser.add_argument('--work-dir', type=Path, default=CORE_DIR / 'runs' / 'no_ortho',
                        help='Parent directory for downloaded and extracted training data')
    parser.add_argument('--data-dir', type=Path,
                        help='Reuse extracted model/style/*.png data instead of downloading shards')
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp32'), default='auto')
    args = parser.parse_args()
    if args.batch_size < 2 or args.epochs < 1:
        parser.error('--batch-size must be at least 2 and --epochs must be positive')
    return args


def pool_shards(work_dir):
    from huggingface_hub import hf_hub_download

    data_dir = work_dir / 'mega_train_data'
    data_dir.mkdir(parents=True)
    data_dir = data_dir.resolve()
    for shard in TARGET_SHARDS:
        name = f'styleid-s-{shard:06d}.tar'
        archive_path = Path(hf_hub_download(
            repo_id='kwanY/stylebench-s', filename=name, repo_type='dataset',
            local_dir=work_dir / 'archives',
        ))
        with tarfile.open(archive_path, 'r') as archive:
            png = None
            for member in tqdm(archive, desc=f'Extract {name}'):
                if not member.isfile():
                    continue
                if member.name.endswith('.png'):
                    with archive.extractfile(member) as image:
                        png = image.read()
                elif member.name.endswith('.json') and png is not None:
                    with archive.extractfile(member) as metadata:
                        original_path = json.load(metadata).get('original_path')
                    if original_path:
                        dest = (data_dir / original_path).resolve()
                        if not dest.is_relative_to(data_dir):
                            raise ValueError(f'Archive path escapes {data_dir}: {original_path}')
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(png)
                    png = None
        archive_path.unlink()
    return data_dir


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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


def apply_lora_to_clip(model, device, r=8):
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
        # Retain this parameter for compatibility with the historical head layout.
        self.ortho_proj = nn.Linear(id_dim, style_dim, bias=False)

    def forward(self, feat):
        z_id = F.normalize(self.id_proj(feat), dim=-1)
        z_style = F.normalize(self.style_proj(feat), dim=-1)
        return z_id, z_style, self.style_classifier(z_style)


class AngularHead(nn.Module):
    def __init__(self, embed_dim, num_classes, margin=0.3, scale=30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)
        self.margin, self.scale = margin, scale

    def forward(self, x, labels=None):
        logits = F.normalize(x, dim=-1) @ F.normalize(self.W, dim=-1).t()
        if labels is None:
            return logits
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
        mask = torch.eq(labels, labels.T).float()
        diagonal = torch.arange(batch_size, device=features.device).view(-1, 1)
        logits_mask = torch.ones_like(mask).scatter_(1, diagonal, 0)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)
        mask_pos = torch.where(mask.sum(1) < 1e-6, 1.0, mask.sum(1))
        return -((mask * log_prob).sum(1) / mask_pos).mean()


def get_features(model, pixels):
    return model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)


class MultiStyleDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.class_to_idx = {}
        self.style_to_idx = {}
        for model_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if not model_dir.is_dir():
                continue
            for style_dir in sorted(model_dir.iterdir(), key=lambda path: path.name):
                if not style_dir.is_dir():
                    continue
                for image in sorted(style_dir.glob('*.png'), key=lambda path: path.name):
                    identity = image.name.split('_')[0]
                    id_label = self.class_to_idx.setdefault(identity, len(self.class_to_idx))
                    style_label = self.style_to_idx.setdefault(style_dir.name, len(self.style_to_idx))
                    self.samples.append((image, id_label, style_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, id_label, style_label = self.samples[idx]
        return load_image(str(image)).convert('RGB'), id_label, style_label


def collate_fn(batch):
    images, id_labels, style_labels = zip(*batch)
    return list(images), torch.tensor(id_labels, dtype=torch.long), torch.tensor(style_labels, dtype=torch.long)


def main(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no CUDA GPU is available')
    supports_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    use_bf16 = args.precision == 'bf16' or (args.precision == 'auto' and supports_bf16)
    if use_bf16 and not supports_bf16:
        raise RuntimeError('BF16 requires a CUDA GPU with BF16 support')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    run_name = 'id_style_no_ortho_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_dir = args.output_dir.resolve() / run_name
    work_dir = args.work_dir.resolve() / run_name
    if args.data_dir is None and (run_dir.is_relative_to(work_dir) or work_dir.is_relative_to(run_dir)):
        raise ValueError('Use separate output and work directories')
    data_dir = args.data_dir.resolve() if args.data_dir else pool_shards(work_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Training data directory is missing: {data_dir}')
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f'Output: {run_dir}', flush=True)

    processor = CLIPProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = CLIPModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION).to(device)
    model.requires_grad_(False)
    apply_lora_to_clip(model, device, r=8)
    head = DisentangledHead(clip_dim=model.config.projection_dim).to(device)
    head.ortho_proj.requires_grad_(False)
    ce_loss = nn.CrossEntropyLoss()
    supcon_loss = SupConLoss().to(device)

    dataset = MultiStyleDataset(data_dir)
    if len(dataset) < args.batch_size:
        raise ValueError('Training data must contain at least one full batch')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                        drop_last=True, num_workers=0, pin_memory=device == 'cuda')
    recipe = dict(
        seed=args.seed, shards=list(TARGET_SHARDS) if args.data_dir is None else None,
        data_dir=str(data_dir), batch_size=args.batch_size, epochs=args.epochs,
        data_order='All images pooled and globally shuffled each epoch',
        initialization='OpenAI CLIP; fresh head and LoRA', base_revision=BASE_REVISION,
        loss='AngularCE + SupCon + 0.5*StyleCE', ortho_weight=0.0,
        optimizer='AdamW', learning_rate=1.2e-4, weight_decay=1e-4,
        scheduler='CosineAnnealingLR', eta_min=1e-6, precision='BF16' if use_bf16 else 'FP32',
        lora_rank=8, lora_alpha=1.0, angular_margin=0.3, angular_scale=30, temperature=0.1,
        checkpoint_selection='Final epoch; no benchmark selection',
        image_count=len(dataset), identity_count=len(dataset.class_to_idx),
        style_count=len(dataset.style_to_idx), class_to_idx=dataset.class_to_idx,
        style_to_idx=dataset.style_to_idx, source_sha256=file_sha256(Path(__file__)),
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy')},
    )
    (run_dir / 'recipe.json').write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding='utf-8')

    angular_head = AngularHead(512, len(dataset.class_to_idx)).to(device)
    params = [
        param for module in (model, head, angular_head)
        for param in module.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(params, lr=1.2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader), eta_min=1e-6,
    )
    for epoch in range(args.epochs):
        model.train()
        head.train()
        batches = tqdm(loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for images, id_labels, style_labels in batches:
            id_labels, style_labels = id_labels.to(device), style_labels.to(device)
            pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16) if use_bf16 else nullcontext():
                features = get_features(model, pixels)
                z_id, _, style_logits = head(features)
                loss_id = ce_loss(angular_head(z_id, id_labels), id_labels) + supcon_loss(z_id, id_labels)
                loss = loss_id + 0.5 * ce_loss(style_logits, style_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            batches.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

    backbone_path = run_dir / 'disentangled_lora_clip_no_ortho.pt'
    head_path = run_dir / 'disentangled_head_no_ortho.pt'
    torch.save(model.state_dict(), backbone_path)
    torch.save(head.state_dict(), head_path)
    completed = dict(recipe, status='complete', checkpoint_sha256={
        path.name: file_sha256(path) for path in (backbone_path, head_path)
    })
    (run_dir / 'completed.json').write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.data_dir is None:
        shutil.rmtree(work_dir)
    print(f'Training complete: {run_dir}', flush=True)


if __name__ == '__main__':
    main(parse_args())
