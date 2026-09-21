import os
import glob
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset: Folder-per-identity
# ============================================================
class StyleIDS_Dataset(Dataset):
    """
    Dataset structure:
    - styleid-s/p3/{id}.png
    - styleid-s/{method}/{style}/{id}_{random}.png

    Identity = filename prefix:
      - p3: full filename without extension
      - others: prefix before first "_"
    """

    def __init__(self, root, processor):
        self.samples = []  # list of (image_path, class_id:int)
        self.processor = processor
        self.class_to_idx = {}

        current_id = 0

        style_methods = sorted(os.listdir(root))  # p3, inf3, inst3, ...

        for method in style_methods:
            method_path = os.path.join(root, method)
            if not os.path.isdir(method_path):
                continue

            # --------------------------------------------------
            # Case 1: p3 (no style subfolders)
            # --------------------------------------------------
            if method == "p3":
                img_paths = glob.glob(os.path.join(method_path, "*.png"))

                for p in img_paths:
                    filename = os.path.basename(p)
                    identity = os.path.splitext(filename)[0]  # class name

                    if identity not in self.class_to_idx:
                        self.class_to_idx[identity] = current_id
                        current_id += 1

                    label = self.class_to_idx[identity]
                    self.samples.append((p, label))

            # --------------------------------------------------
            # Case 2: stylized images
            # --------------------------------------------------
            else:
                style_categories = sorted(os.listdir(method_path))

                for cat in style_categories:
                    cat_path = os.path.join(method_path, cat)
                    if not os.path.isdir(cat_path):
                        continue

                    img_paths = glob.glob(os.path.join(cat_path, "*.png"))

                    for p in img_paths:
                        filename = os.path.basename(p)
                        identity = filename.split("_")[0]

                        if identity not in self.class_to_idx:
                            self.class_to_idx[identity] = current_id
                            current_id += 1

                        label = self.class_to_idx[identity]
                        self.samples.append((p, label))

        print(
            f"Loaded {len(self.samples)} samples "
            f"from {len(self.class_to_idx)} identities "
            f"(including p3 source images)."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_image(path).convert("RGB")
        return img, label



# ============================================================
# CLIP Model + Processor
# ============================================================
ckpt = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(ckpt).to(device)
processor = CLIPProcessor.from_pretrained(ckpt)

# Freeze CLIP weights
for p in model.parameters():
    p.requires_grad = False

def collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels


# ============================================================
# LoRA Module
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.scaling = alpha / r

        self.lora_down = nn.Linear(linear_layer.in_features, r, bias=False)
        self.lora_up   = nn.Linear(r, linear_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling


# ============================================================
# Apply LoRA to SigLIP Self-Attention layers
# ============================================================
def apply_lora_to_clip(model, r=8):
    layers = model.vision_model.encoder.layers
    for layer in layers:
        attn = layer.self_attn
        attn.q_proj = LoRALinear(attn.q_proj, r=r).to(device)
        attn.k_proj = LoRALinear(attn.k_proj, r=r).to(device)
        attn.v_proj = LoRALinear(attn.v_proj, r=r).to(device)
        attn.out_proj = LoRALinear(attn.out_proj, r=r).to(device)

apply_lora_to_clip(model, r=8)


# ============================================================
# Angular classification head (ArcFace-like)
# ============================================================
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

        theta = torch.acos(logits.clamp(-1+1e-5, 1-1e-5))
        target_logits = torch.cos(theta + self.margin)

        onehot = torch.zeros_like(logits)
        onehot.scatter_(1, labels.unsqueeze(1), 1)

        logits = logits * (1 - onehot) + target_logits * onehot
        logits *= self.scale
        return logits


# ============================================================
# Loss Functions
# ============================================================
ce_loss = nn.CrossEntropyLoss()

def contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return ce_loss(logits, labels)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        # Shape: [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T)

        # Create the mask for positive pairs (same label)
        # mask[i, j] = 1 if labels[i] == labels[j]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask-out self-contrast (diagonal) because distance to self is always 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute logits
        # subtract max for numerical stability
        logits = similarity_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positive
        # Avoid division by zero if an image has no other positive in the batch
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss

supcon_loss_fn = SupConLoss(temperature=0.1).to(device)

@torch.no_grad()
def get_clip_baseline_emb(inputs):
    return F.normalize(model.get_image_features(**inputs), dim=-1)

def align_loss(new_emb, base_emb):
    new_emb = F.normalize(new_emb, dim=-1)
    base_emb = F.normalize(base_emb, dim=-1)
    return 1 - (new_emb * base_emb).sum(dim=-1).mean()


# ============================================================
# Training Step
# ============================================================
def training_step(images, labels, angular_head, optimizer):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        base_emb = get_clip_baseline_emb(inputs)

    new_emb = model.get_image_features(**inputs)

    logits = angular_head(new_emb, labels)
    loss_ang = ce_loss(logits, labels)

    #loss_contra = contrastive_loss(new_emb, new_emb.detach())
    loss_contra = supcon_loss_fn(new_emb, labels)
    loss_align = align_loss(new_emb, base_emb)

    loss = 1.0 * loss_ang + loss_contra + 0.1 * loss_align

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wandb.log({
        "loss/total": loss.item(),
        "loss/angular": loss_ang.item(),
        "loss/contrastive": loss_contra.item(),
    })

    return loss.item()


# ============================================================
# Full Training Loop with wandb
# ============================================================
def train(model, dataset_root, epochs=5, batch_size=48, lr=1e-4):

    # Setup dataset
    dataset = StyleIDS_Dataset(dataset_root, processor)
    num_classes = len(dataset.class_to_idx)

    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn, 
    pin_memory=True,
    persistent_workers=True
    )

    # Build angular classifier
    embed_dim = embed_dim = model.config.projection_dim

    angular_head = AngularHead(embed_dim, num_classes).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(angular_head.parameters())
    num_trainable = sum(p.numel() for p in trainable_params)
    #print(f"{num_trainable/1000000:.3}M")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Initialize wandb
    wandb.init(
        project="clip-identity-finetune",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dataset": dataset_root,
            "num_classes": num_classes,
            "lora_rank": 8,
            "loss_weights": {"angular": 1.0, "contrast": 0.5, "align": 0.1},
        }
    )

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in tqdm.tqdm(dataloader):
            labels = labels.to(device)
            loss = training_step(images, labels, angular_head, optimizer)
            total_loss += loss

        avg_loss = total_loss / len(dataloader)

        # wandb logging
        wandb.log({"epoch": epoch, "epoch_loss": avg_loss})
        print(f"[Epoch {epoch+1}/{epochs}] Loss = {avg_loss:.4f}")

        # Save a checkpoint
        torch.save(
            {
                "angular_head": angular_head.state_dict(),
                "model": model.state_dict(),
            },
            f"ckpt/checkpoint_clip_epoch{epoch+1}.pt"
        )
        wandb.save(f"checkpoint_clip_epoch{epoch+1}.pt")

train(model, dataset_root="styleid-s/", epochs=5, batch_size=56, lr=1e-4)
