# Compare Models
import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('--> Thiet bi:', device.upper())
DATA_ROOT = 'styleid-s'

class MultiStyleDataset(Dataset):
    def __init__(self, root, is_train=False, val_split=0.2, seed=42):
        self.samples, self.class_to_idx, self.style_to_idx, all_samples = [], {}, {}, []
        for m in sorted(os.listdir(root)):
            mp = os.path.join(root, m)
            if not os.path.isdir(mp): continue
            for cat in sorted(os.listdir(mp)):
                cp = os.path.join(mp, cat)
                if not os.path.isdir(cp): continue
                for p in glob.glob(os.path.join(cp, '*.png')):
                    all_samples.append((p, os.path.basename(p).split('_')[0], cat))
        u_ids = sorted(list(set([s[1] for s in all_samples])))
        random.seed(seed); random.shuffle(u_ids)
        split_idx = int(len(u_ids) * (1 - val_split))
        target_ids = set(u_ids[:split_idx]) if is_train else set(u_ids[split_idx:])
        for idx, sty in enumerate(sorted(list(set([s[2] for s in all_samples])))):
            self.style_to_idx[sty] = idx
        for path, ident, style in all_samples:
            if ident in target_ids:
                if ident not in self.class_to_idx: self.class_to_idx[ident] = len(self.class_to_idx)
                self.samples.append((path, self.class_to_idx[ident], self.style_to_idx[style], ident, style))
        print(f'Test set: {len(self.samples)} anh | {len(self.class_to_idx)} IDs')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, id_lbl, sty_lbl, _, _ = self.samples[idx]
        return load_image(p).convert('RGB'), id_lbl, sty_lbl

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
    def __init__(self, clip_dim=768, id_dim=512, style_dim=256, num_styles=3):
        super().__init__()
        self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
        self.style_proj = nn.Sequential(nn.Linear(clip_dim, style_dim), nn.BatchNorm1d(style_dim), nn.PReLU())
        self.style_classifier = nn.Linear(style_dim, max(num_styles, 2))
        self.ortho_proj = nn.Linear(id_dim, style_dim, bias=False)
    def forward(self, feat):
        return F.normalize(self.id_proj(feat), dim=-1), F.normalize(self.style_proj(feat), dim=-1), self.style_classifier(F.normalize(self.style_proj(feat), dim=-1))

def get_features(m, pv):
    return m.visual_projection(m.vision_model(pixel_values=pv).pooler_output)

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
    return mean_cross, mean_neg, cross_acc, overall_acc

def main():
    val_dataset = MultiStyleDataset(DATA_ROOT, is_train=False, val_split=0.2, seed=42)
    print('\n[1/2] Loading Disentangled model from checkpoint...')
    my_proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    my_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    for p in my_model.parameters(): p.requires_grad = False
    apply_lora_to_clip(my_model, r=8)
    my_model.load_state_dict(torch.load('disentangled_lora_clip.pt', map_location=device))
    my_model.eval()
    head = DisentangledHead(clip_dim=my_model.config.projection_dim, id_dim=512, style_dim=256, num_styles=len(val_dataset.style_to_idx)).to(device)
    head.load_state_dict(torch.load('disentangled_head.pt', map_location=device))
    head.eval()

    print('\n[2/2] Loading Official StyleID (kwanY/styleid - Full 135GB)...')
    off_proc = CLIPProcessor.from_pretrained('kwanY/styleid')
    off_model = CLIPModel.from_pretrained('kwanY/styleid').to(device)
    off_model.eval()

    print('\n--> Extracting features for both models on same 500 test images...')
    my_embs, off_embs, id_lbls, sty_lbls = [], [], [], []
    with torch.no_grad():
        for i in tqdm.tqdm(range(min(len(val_dataset), 500))):
            img, id_lbl, sty_lbl = val_dataset[i]
            pv = my_proc(images=img, return_tensors='pt')['pixel_values'].to(device)
            z_id, _, _ = head(get_features(my_model, pv))
            my_embs.append(z_id)

            off_pv = off_proc(images=img, return_tensors='pt')['pixel_values'].to(device)
            off_feat = F.normalize(get_features(off_model, off_pv), dim=-1)
            off_embs.append(off_feat)

            id_lbls.append(id_lbl)
            sty_lbls.append(sty_lbl)

    my_embs = torch.cat(my_embs, dim=0)
    off_embs = torch.cat(off_embs, dim=0)

    my_cross, my_neg, my_c_acc, my_o_acc = evaluate_cross_style(my_embs, id_lbls, sty_lbls)
    off_cross, off_neg, off_c_acc, off_o_acc = evaluate_cross_style(off_embs, id_lbls, sty_lbls)

    print('\n' + '='*75)
    print('   BANG DOI DAU TRUC DIEN: DISENTANGLED (CUA BAN) VS STYLEID (CHINH CHU)')
    print('='*75)
    print(f'Tieu chi                            | Disentangled (Ban) | StyleID (Chinh chu)')
    print('-'*75)
    print(f'Do chinh xac Cross-Style (Khac style) | {my_c_acc*100:<18.2f}% | {off_c_acc*100:<18.2f}%')
    print(f'Do chinh xac Tong the (Overall Acc)   | {my_o_acc*100:<18.2f}% | {off_o_acc*100:<18.2f}%')
    print(f'Do tuong dong CUNG NGUOI (Khac style) | {my_cross:<19.4f} | {off_cross:<19.4f}')
    print(f'Do tuong dong KHAC NGUOI              | {my_neg:<19.4f} | {off_neg:<19.4f}')
    print(f'Khoang cach phan tach (Margin Gap)    | {my_cross-my_neg:<19.4f} | {off_cross-off_neg:<19.4f}')
    print('='*75 + '\n')

if __name__ == '__main__':
    main()
