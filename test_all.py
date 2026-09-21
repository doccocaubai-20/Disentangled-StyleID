# ==============================================================================
# ALL-IN-ONE BENCHMARK: DREAM SOUP vs SHARD 6 vs A100 50% vs STYLEID
# TỰ ĐỘNG KẾT NỐI DRIVE + TỰ TẢI SKSF-A + ĐÁNH GIÁ CẢ TABLE 2 & TABLE 9
# ==============================================================================
import os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image
import tqdm, subprocess, zipfile, shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 THIẾT BỊ: {device.upper()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ------------------------------------------------------------------------------
# 0. TỰ ĐỘNG KẾT NỐI GOOGLE DRIVE
# ------------------------------------------------------------------------------
if not os.path.exists('/content/drive'):
    from google.colab import drive
    print("\n--> [BƯỚC 0/5] Đang kết nối Google Drive...")
    drive.mount('/content/drive')
print("--> ✅ Google Drive đã sẵn sàng!")

# ------------------------------------------------------------------------------
# 1. TỰ ĐỘNG TẢI & GIẢI NÉN SKSF-A NẾU CHƯA CÓ
# ------------------------------------------------------------------------------
print("\n--> [BƯỚC 1/5] Chuẩn bị dữ liệu kiểm thử SKSF-A (Tranh nghệ thuật)...")

def check_sksf_exists():
    if not os.path.exists('SKSF-A_data'): return False
    for root, _, files in os.walk('SKSF-A_data'):
        if 'sksf_a.csv' in files: return True
    return False

if not check_sksf_exists():
    print("  + Chưa có dữ liệu SKSF-A, đang tự động nạp...")
    drive_zip = '/content/drive/MyDrive/SKSF-A.zip'
    if not os.path.exists('SKSF-A.zip'):
        if os.path.exists(drive_zip):
            print("  + Sao chép SKSF-A.zip từ Drive...")
            shutil.copy(drive_zip, 'SKSF-A.zip')
        else:
            print("  + Tải nhanh SKSF-A.zip qua gdown...")
            os.system('pip install -q gdown')
            os.system('gdown 1E0QBScC_0cXoMNn7JE1P4f6I9qMFagvz -O SKSF-A.zip')
            
    if os.path.exists('SKSF-A.zip'):
        print("  + Đang giải nén SKSF-A.zip...")
        with zipfile.ZipFile('SKSF-A.zip', 'r') as zf:
            zf.extractall('SKSF-A_data')
        print("  + ✅ Giải nén hoàn tất!")

# Định vị thư mục chứa file csv
sksf_dir = 'SKSF-A_data'
for root, _, files in os.walk('SKSF-A_data'):
    if 'sksf_a.csv' in files:
        sksf_dir = root
        break

csv_path = os.path.join(sksf_dir, 'sksf_a.csv')
print(f"  + Thư mục làm việc SKSF-A: {sksf_dir}")

def find_file(f, idx):
    for ext in ['.PNG', '.png', '.jpg', '.jpeg', '.JPG']:
        p = os.path.join(f, f"{idx}{ext}")
        if os.path.exists(p): return p
    return None

df = pd.read_csv(csv_path)
image_ids = df['Image_no'].tolist()
styles = [f"style{i}" for i in range(1, 8)]

pairs_pos, pairs_neg = [], []
random.seed(42)
for img_id in image_ids:
    photo_p = find_file(os.path.join(sksf_dir, 'Photo'), img_id)
    if not photo_p: continue
    for st in styles:
        sk_p = find_file(os.path.join(sksf_dir, st), img_id)
        if sk_p:
            pairs_pos.append((photo_p, sk_p, 1))
            other = random.choice([o for o in image_ids if o != img_id])
            neg_sk_p = find_file(os.path.join(sksf_dir, st), other)
            if neg_sk_p: pairs_neg.append((photo_p, neg_sk_p, 0))

sksf_pairs = pairs_pos + pairs_neg
y_sksf = np.array([p[2] for p in sksf_pairs])
sksf_unique_paths = list(set([p[0] for p in sksf_pairs] + [p[1] for p in sksf_pairs]))
print(f"  + ✅ SKSF-A: {len(sksf_pairs)} cặp test ({len(pairs_pos)} Cùng người / {len(pairs_neg)} Khác người) | {len(sksf_unique_paths)} ảnh")

# ------------------------------------------------------------------------------
# 2. TỰ ĐỘNG NẠP DỮ LIỆU LFW (TABLE 9)
# ------------------------------------------------------------------------------
print("\n--> [BƯỚC 2/5] Chuẩn bị dữ liệu kiểm thử LFW (Mặt người thật)...")
lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0)
lfw_pairs_raw = lfw.pairs
y_lfw = lfw.target

lfw_imgs1, lfw_imgs2 = [], []
for i in range(len(lfw_pairs_raw)):
    p1 = (lfw_pairs_raw[i, 0] * 255).astype(np.uint8) if lfw_pairs_raw[i, 0].max() <= 1.0 else lfw_pairs_raw[i, 0].astype(np.uint8)
    p2 = (lfw_pairs_raw[i, 1] * 255).astype(np.uint8) if lfw_pairs_raw[i, 1].max() <= 1.0 else lfw_pairs_raw[i, 1].astype(np.uint8)
    lfw_imgs1.append(Image.fromarray(p1).convert('RGB'))
    lfw_imgs2.append(Image.fromarray(p2).convert('RGB'))

print(f"  + ✅ LFW: {len(lfw_pairs_raw)} cặp test ({np.sum(y_lfw==1)} Cùng người / {np.sum(y_lfw==0)} Khác người)")

# ------------------------------------------------------------------------------
# 3. ĐỊNH NGHĨA KIẾN TRÚC & HÀM ĐÁNH GIÁ BATCH SIÊU TỐC
# ------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=1.0):
        super().__init__()
        self.linear, self.scaling = linear_layer, alpha / r
        self.lora_down = nn.Linear(linear_layer.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, linear_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
    def forward(self, x): return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling

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
    def forward(self, feat): return F.normalize(self.id_proj(feat), dim=-1)

def get_features(m, pv): return m.visual_projection(m.vision_model(pixel_values=pv).pooler_output)

processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

def load_image_clean(p):
    img = Image.open(p)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        c = Image.new('RGB', img.size, (255, 255, 255))
        c.paste(img, mask=img.split()[3])
        return c
    return img.convert('RGB')

def calc_sksf_metrics(y, s):
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    idx_1fpr = np.where(fpr <= 0.01)[0]
    tpr1 = tpr[idx_1fpr[-1]] if len(idx_1fpr) > 0 else 0.0
    pm, nm = (y == 1), (y == 0)
    a3 = (np.sum(s[pm] >= 0.3) + np.sum(s[nm] < 0.3)) / len(y)
    a4 = (np.sum(s[pm] >= 0.4) + np.sum(s[nm] < 0.4)) / len(y)
    a5 = (np.sum(s[pm] >= 0.5) + np.sum(s[nm] < 0.5)) / len(y)
    return tpr1, a3, a4, a5, auc, np.mean(s[pm]), np.mean(s[nm])

def calc_lfw_metrics(y, s):
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    idx_1fpr = np.where(fpr <= 0.01)[0]
    tpr1 = tpr[idx_1fpr[-1]] if len(idx_1fpr) > 0 else 0.0
    pm, nm = (y == 1), (y == 0)
    a3 = (np.sum(s[pm] >= 0.3) + np.sum(s[nm] < 0.3)) / len(y)
    best_acc, best_th = 0.0, 0.0
    for th in np.linspace(-0.2, 0.95, 250):
        acc = (np.sum(s[pm] >= th) + np.sum(s[nm] < th)) / len(y)
        if acc > best_acc: best_acc, best_th = acc, th
    return tpr1, a3, auc, best_acc, best_th, np.mean(s[pm]), np.mean(s[nm])

def evaluate_model_on_both(model, head, is_official=False):
    model.eval()
    if head is not None: head.eval()
    
    # 1. SKSF-A Caching Batch
    cache = {}
    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(sksf_unique_paths), BATCH):
            bp = sksf_unique_paths[i:i+BATCH]
            b_imgs = [load_image_clean(p) for p in bp]
            pv = processor(images=b_imgs, return_tensors='pt')['pixel_values'].to(device)
            feats = get_features(model, pv)
            embs = F.normalize(feats, dim=-1) if is_official else head(feats)
            for p, emb in zip(bp, embs): cache[p] = emb
            
    sc_sksf = [(cache[p1] * cache[p2]).sum(dim=-1).item() for p1, p2, _ in sksf_pairs]
    
    # 2. LFW Batch
    sc_lfw = []
    with torch.no_grad():
        for i in range(0, len(lfw_imgs1), BATCH):
            b1 = lfw_imgs1[i:i+BATCH]
            b2 = lfw_imgs2[i:i+BATCH]
            pv1 = processor(images=b1, return_tensors='pt')['pixel_values'].to(device)
            pv2 = processor(images=b2, return_tensors='pt')['pixel_values'].to(device)
            f1, f2 = get_features(model, pv1), get_features(model, pv2)
            if is_official:
                e1, e2 = F.normalize(f1, dim=-1), F.normalize(f2, dim=-1)
            else:
                e1, e2 = head(f1), head(f2)
            sims = (e1 * e2).sum(dim=-1).cpu().numpy().tolist()
            sc_lfw.extend(sims)
            
    return np.array(sc_sksf), np.array(sc_lfw)

# ------------------------------------------------------------------------------
# 4. NẠP TRỌNG SỐ CÁC CHECKPOINT
# ------------------------------------------------------------------------------
print("\n--> [BƯỚC 3/5] Nạp các Checkpoint...")
SAVE_DIR = '/content/drive/MyDrive/StyleID_A100_Champion'

def find_ckpt(fname):
    for d in [SAVE_DIR, '/content/drive/MyDrive', '.']:
        p = os.path.join(d, fname)
        if os.path.exists(p): return p
    return None

# Shard 6
s6_l_path = find_ckpt('disentangled_lora_clip.pt')
s6_h_path = find_ckpt('disentangled_head.pt')
print(f"  + Shard 6: {s6_l_path}")

# A100 50%
a100_l_path = find_ckpt('disentangled_lora_clip_a100_final.pt')
a100_h_path = find_ckpt('disentangled_head_a100_final.pt')
print(f"  + A100 50%: {a100_l_path}")

# Dream Soup
soup_l_path = find_ckpt('disentangled_lora_clip_SOUP_S6_A100.pt')
soup_h_path = find_ckpt('disentangled_head_SOUP_S6_A100.pt')

sd_s6_l = torch.load(s6_l_path, map_location='cpu')
sd_s6_h = torch.load(s6_h_path, map_location='cpu')
sd_a100_l = torch.load(a100_l_path, map_location='cpu')
sd_a100_h = torch.load(a100_h_path, map_location='cpu')

if soup_l_path and os.path.exists(soup_l_path):
    print(f"  + DREAM SOUP: {soup_l_path} (Đã có sẵn)")
    sd_soup_l = torch.load(soup_l_path, map_location='cpu')
    sd_soup_h = torch.load(soup_h_path, map_location='cpu')
else:
    print("  + DREAM SOUP: Đang hòa trộn 50/50 trên RAM...")
    sd_soup_l = {k: (0.5*sd_s6_l[k].float() + 0.5*sd_a100_l[k].float()).to(sd_a100_l[k].dtype) for k in sd_a100_l}
    sd_soup_h = {k: (0.5*sd_s6_h[k].float() + 0.5*sd_a100_h[k].float()).to(sd_a100_h[k].dtype) for k in sd_a100_h}

# ------------------------------------------------------------------------------
# 5. CHẤM ĐIỂM SIÊU TỐC CẢ 4 MÔ HÌNH (~30 GIÂY)
# ------------------------------------------------------------------------------
print("\n--> [BƯỚC 4/5] BẮT ĐẦU CHẤM ĐIỂM ĐỐI ĐẦU CHÍNH THỨC...")
results_sksf, results_lfw = {}, {}

# A. StyleID Chính chủ
print("  --> [1/4] Chấm điểm StyleID Chính chủ...")
off_model = CLIPModel.from_pretrained("kwanY/styleid").to(device)
sc_sksf_o, sc_lfw_o = evaluate_model_on_both(off_model, None, is_official=True)
results_sksf['StyleID (Chính chủ)'] = calc_sksf_metrics(y_sksf, sc_sksf_o)
results_lfw['StyleID (Chính chủ)'] = calc_lfw_metrics(y_lfw, sc_lfw_o)
del off_model; torch.cuda.empty_cache()

# B. Disentangled Models
dis_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
for p in dis_model.parameters(): p.requires_grad = False
apply_lora_to_clip(dis_model, r=8)
dis_head = DisentangledHead().to(device)

def test_dis_weights(name, w_l, w_h):
    print(f"  --> Chấm điểm {name}...")
    dis_model.load_state_dict(w_l, strict=False)
    cur_h = dis_head.state_dict()
    for k, v in w_h.items():
        if k in cur_h and cur_h[k].shape == v.shape: cur_h[k] = v.to(device)
    dis_head.load_state_dict(cur_h)
    
    sc_s, sc_l = evaluate_model_on_both(dis_model, dis_head, is_official=False)
    results_sksf[name] = calc_sksf_metrics(y_sksf, sc_s)
    results_lfw[name] = calc_lfw_metrics(y_lfw, sc_l)

# B1. Bản Shard 6
test_dis_weights('Disentangled (Bản Shard 6)', sd_s6_l, sd_s6_h)

# B2. Bản A100 50%
test_dis_weights('Disentangled (Bản A100 50%)', sd_a100_l, sd_a100_h)

# B3. DREAM SOUP 🍲
test_dis_weights('Disentangled (DREAM SOUP 🍲)', sd_soup_l, sd_soup_h)

# ------------------------------------------------------------------------------
# 6. IN 2 BẢNG THÀNH TÍCH CHUẨN KHOA HỌC
# ------------------------------------------------------------------------------
print('\n' + '='*96)
print('BẢNG 1: ĐỐI ĐẦU TRÊN SKSF-A (TABLE 2: STYLIZED FACE VERIFICATION - 1.876 CẶP)')
print('='*96)
print(f"{'Method':<30} | {'TPR↑ (@1%)':<11} | {'Acc@0.3↑':<9} | {'Acc@0.4↑':<9} | {'Acc@0.5↑':<9} | {'AUROC↑':<8} | {'Pos Sim↑':<8}")
print('-'*96)
print(f"{'ArcFace (Bài báo công bố)':<30} | 0.6189      | 0.6801    | 0.5346    | 0.5059    | 0.9360   | —")
print(f"{'AdaFace (Bài báo công bố)':<30} | 0.6993      | 0.6178    | 0.5357    | 0.5032    | 0.9582   | —")
print(f"{'CLIP (Bài báo công bố)':<30} | 0.4968      | 0.5000    | 0.5059    | 0.5778    | 0.9420   | —")
print(f"{'StyleID (Bài báo công bố)':<30} | 0.8891      | 0.8731    | 0.7393    | 0.6178    | 0.9922   | —")
print('-'*96)
for name, m in results_sksf.items():
    print(f"{name:<30} | {m[0]:<11.4f} | {m[1]:<9.4f} | {m[2]:<9.4f} | {m[3]:<9.4f} | {m[4]:<8.4f} | {m[5]:<8.4f}")
print('='*96)

print('\n' + '='*96)
print('BẢNG 2: ĐỐI ĐẦU TRÊN LFW (TABLE 9: NATURAL-FACE VERIFICATION - 1.000 CẶP)')
print('='*96)
print(f"{'Method':<30} | {'TPR↑ (@1% FPR)':<16} | {'AUC↑':<10} | {'Best Acc↑':<18} | {'Pos Sim↑':<10} | {'Margin↑':<10}")
print('-'*96)
print(f"{'ArcFace (Bài báo công bố)':<30} | 0.9970           | 0.9989     | —                  | —          | —")
print(f"{'StyleID (Bài báo công bố)':<30} | 0.9526           | 0.9967     | —                  | —          | —")
print('-'*96)
for name, m in results_lfw.items():
    best_str = f"{m[3]*100:.2f}% (th={m[4]:.2f})"
    print(f"{name:<30} | {m[0]:<16.4f} | {m[2]:<10.4f} | {best_str:<18} | {m[5]:<10.4f} | {m[5]-m[6]:<10.4f}")
print('='*96)

print("\n🎉 HOÀN TẤT ĐÁNH GIÁ ĐỒNG THỜI TOÀN DIỆN CẢ 4 MÔ HÌNH TRÊN 2 BÀI THI QUỐC TẾ!")