import sys
# Đảm bảo in tiếng Việt và emoji trên console Windows không bị lỗi cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os, glob, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Hàm load ảnh chuẩn: Xử lý tranh vẽ tay trong suốt (RGBA) lên nền giấy trắng
def load_image_clean(path):
    img = Image.open(path)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        canvas = Image.new('RGB', img.size, (255, 255, 255))
        canvas.paste(img, mask=img.split()[3])
        return canvas
    return img.convert('RGB')

def find_file(folder, img_id):
    for ext in ['.PNG', '.png', '.jpg', '.jpeg', '.JPG']:
        p = os.path.join(folder, f"{img_id}{ext}")
        if os.path.exists(p): return p
    return None

# 2. Xây dựng danh sách 938 cặp Photo-to-Sketch (Chuẩn 100% Bảng 2)
df = pd.read_csv('SKSF-A_data/sksf_a.csv')
image_ids = df['Image_no'].tolist()
styles = [f"style{i}" for i in range(1, 8)]

pairs_pos = []  # (photo_path, sketch_path, 1)
pairs_neg = []  # (photo_path, sketch_path, 0)

random.seed(42)
for img_id in image_ids:
    photo_p = find_file('SKSF-A_data/Photo', img_id)
    if not photo_p: continue
    
    # Tạo các cặp cùng người (Positive: Photo_i <-> Sketch_i các style)
    for st in styles:
        sketch_p = find_file(f'SKSF-A_data/{st}', img_id)
        if sketch_p:
            pairs_pos.append((photo_p, sketch_p, 1))
            
            # Tạo 1 cặp khác người tương ứng (Negative: Photo_i <-> Sketch_j cùng style)
            other_ids = [o for o in image_ids if o != img_id]
            neg_id = random.choice(other_ids)
            neg_sketch_p = find_file(f'SKSF-A_data/{st}', neg_id)
            if neg_sketch_p:
                pairs_neg.append((photo_p, neg_sketch_p, 0))

print(f"--> ✅ ĐÃ TẠO BÀI THI PHOTO-TO-SKETCH CHUẨN:")
print(f"    + Số cặp cùng người (Positive Pairs): {len(pairs_pos)}")
print(f"    + Số cặp khác người (Negative Pairs): {len(pairs_neg)}")
eval_pairs = pairs_pos + pairs_neg

# 3. Định nghĩa kiến trúc Disentangled
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
    def __init__(self, clip_dim=768, id_dim=512, style_dim=256):
        super().__init__()
        self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
        self.style_proj = nn.Sequential(nn.Linear(clip_dim, style_dim), nn.BatchNorm1d(style_dim), nn.PReLU())
    def forward(self, feat):
        return F.normalize(self.id_proj(feat), dim=-1)

def get_features(m, pv):
    return m.visual_projection(m.vision_model(pixel_values=pv).pooler_output)

# 4. Nạp 2 mô hình (Bản Shard 6 & StyleID chính chủ)
print("\n--> [1/3] Nạp mô hình Disentangled BẢN SHARD 6...")
my_proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
my_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
for p in my_model.parameters(): p.requires_grad = False
apply_lora_to_clip(my_model, r=8)
my_model.load_state_dict(torch.load('disentangled_lora_clip.pt', map_location=device), strict=False)
my_model.eval()

my_head = DisentangledHead(clip_dim=my_model.config.projection_dim, id_dim=512, style_dim=256).to(device)
old_st = torch.load('disentangled_head.pt', map_location=device)
cur_st = my_head.state_dict()
for k, v in old_st.items():
    if k in cur_st and cur_st[k].shape == v.shape: cur_st[k] = v
my_head.load_state_dict(cur_st)
my_head.eval()

print("\n--> [2/3] Nạp mô hình StyleID chính chủ (kwanY/styleid)...")
off_proc = CLIPProcessor.from_pretrained('kwanY/styleid')
off_model = CLIPModel.from_pretrained('kwanY/styleid').to(device)
off_model.eval()

# 5. Trích xuất đặc trưng và tính độ tương đồng từng cặp
print("\n--> [3/3] Đang chấm điểm 100% chuẩn bài thi Table 2...")
# Bộ nhớ đệm cache embedding để chạy cực nhanh
cache_my = {}
cache_off = {}

def get_my_emb(p):
    if p not in cache_my:
        img = load_image_clean(p)
        pv = my_proc(images=img, return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad():
            cache_my[p] = my_head(get_features(my_model, pv))
    return cache_my[p]

def get_off_emb(p):
    if p not in cache_off:
        img = load_image_clean(p)
        pv = off_proc(images=img, return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad():
            cache_off[p] = F.normalize(get_features(off_model, pv), dim=-1)
    return cache_off[p]

y_true, scores_my, scores_off = [], [], []

for p1, p2, label in tqdm.tqdm(eval_pairs, desc="Chấm điểm SKSF-A"):
    emb1_my, emb2_my = get_my_emb(p1), get_my_emb(p2)
    emb1_off, emb2_off = get_off_emb(p1), get_off_emb(p2)
    
    sim_my = (emb1_my @ emb2_my.T).item()
    sim_off = (emb1_off @ emb2_off.T).item()
    
    y_true.append(label)
    scores_my.append(sim_my)
    scores_off.append(sim_off)

y_true = np.array(y_true)
scores_my = np.array(scores_my)
scores_off = np.array(scores_off)

# 6. Tính toán chuẩn 5 chỉ số Table 2
def compute_table2_metrics(y_true, scores):
    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(fpr <= 0.01)[0][-1]
    tpr_at_1pct = tpr[idx]
    
    pos_m, neg_m = (y_true == 1), (y_true == 0)
    acc_03 = (np.sum(scores[pos_m] >= 0.3) + np.sum(scores[neg_m] < 0.3)) / len(y_true)
    acc_04 = (np.sum(scores[pos_m] >= 0.4) + np.sum(scores[neg_m] < 0.4)) / len(y_true)
    acc_05 = (np.sum(scores[pos_m] >= 0.5) + np.sum(scores[neg_m] < 0.5)) / len(y_true)
    
    mean_pos = np.mean(scores[pos_m])
    mean_neg = np.mean(scores[neg_m])
    
    return tpr_at_1pct, acc_03, acc_04, acc_05, auroc, mean_pos, mean_neg

tpr_my, a03_my, a04_my, a05_my, auc_my, pos_my, neg_my = compute_table2_metrics(y_true, scores_my)
tpr_off, a03_off, a04_off, a05_off, auc_off, pos_off, neg_off = compute_table2_metrics(y_true, scores_off)

# 7. IN BẢNG ĐỐI ĐẦU CHUẨN XÁC THEO FORMAT BÀNG 2 BÀI BÁO GỐC
print('\n' + '='*85)
print('Table 2. Comparison with baselines on SKSF-A (PHOTO-TO-SKETCH CHUẨN BÀI BÁO)')
print('='*85)
print(f"{'Methods':<28} | {'TPR↑':<8} | {'Acc@0.3↑':<9} | {'Acc@0.4↑':<9} | {'Acc@0.5↑':<9} | {'AUROC↑':<8}")
print('-'*85)
print(f"{'ArcFace (Bài báo công bố)':<28} | 0.6189   | 0.6801    | 0.5346    | 0.5059    | 0.9360")
print(f"{'AdaFace (Bài báo công bố)':<28} | 0.6993   | 0.6178    | 0.5357    | 0.5032    | 0.9582")
print(f"{'CLIP (Bài báo công bố)':<28} | 0.4968   | 0.5000    | 0.5059    | 0.5778    | 0.9420")
print(f"{'StyleID (Bài báo công bố)':<28} | 0.8891   | 0.8731    | 0.7393    | 0.6178    | 0.9922")
print('-'*85)
print(f"{'StyleID (Chính chủ test lại)':<28} | {tpr_off:<8.4f} | {a03_off:<9.4f} | {a04_off:<9.4f} | {a05_off:<9.4f} | {auc_off:<8.4f}")
print(f"{'Disentangled (Bản Shard 6)':<28} | {tpr_my:<8.4f} | {a03_my:<9.4f} | {a04_my:<9.4f} | {a05_my:<9.4f} | {auc_my:<8.4f}")
print('='*85)
print(f"\n[Chi tiết phân bố điểm tương đồng Cosine]:")
print(f"- Cùng người (Pos Sim) ↑: Bản của bạn: {pos_my:.4f} | StyleID chính chủ: {pos_off:.4f}")
print(f"- Khác người (Neg Sim) ↓: Bản của bạn: {neg_my:.4f} | StyleID chính chủ: {neg_off:.4f}")
print(f"- Khoảng cách biên (Margin) ↑: Bản của bạn: {pos_my - neg_my:.4f} | StyleID chính chủ: {pos_off - neg_off:.4f}")