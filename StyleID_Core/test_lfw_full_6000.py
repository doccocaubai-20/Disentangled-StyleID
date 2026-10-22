# Paths work independently of the terminal's current directory.
from pathlib import Path as _Path
import os as _os
_HERE = _Path(__file__).resolve().parent
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_HERE / '.env', override=False)
_DEFAULT_RESOURCES = _HERE if (_HERE/'SKSF-A_data').is_dir() else _HERE.parent
RESOURCE_ROOT = _Path(_os.environ.get('STYLEID_RESOURCES', str(_DEFAULT_RESOURCES))).resolve()
_CKPT_OPTIONS = [_HERE/'checkpoints'/'soup', _HERE, RESOURCE_ROOT]
_DEFAULT_CKPT = next((p for p in _CKPT_OPTIONS if (p/'disentangled_lora_clip_SOUP_S6_A100.pt').is_file()), RESOURCE_ROOT)
CHECKPOINT_ROOT = _Path(_os.environ.get('STYLEID_CHECKPOINT_DIR', str(_DEFAULT_CKPT))).resolve()
_os.chdir(RESOURCE_ROOT)

# ==============================================================================
# ĐÁNH GIÁ CHUẨN QUỐC TẾ LFW (TABLE 9) - TÁI LẬP KẾT QUẢ BÀI BÁO STYLEID
# Giao thức: Huang et al. 2008 (6.000 cặp ảnh - 10 Folds, Tỷ lệ ảnh vuông 1:1)
# Chạy mượt mà 100% trên Notebook mới tinh (Trống trơn), KHÔNG tràn RAM!
# ==============================================================================
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
import tqdm

# Tự động cài đặt transformers nếu notebook mới chưa có
try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    print("--> Đang cài đặt transformers...")
    os.system("pip install -q transformers")
    from transformers import CLIPModel, CLIPProcessor

# 1. THIẾT BỊ TÍNH TOÁN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f"--> [1/4] Thiết bị tính toán: {device.upper()} ({gpu_name})")

# ------------------------------------------------------------------------------
# 2. TỰ ĐỘNG NẠP VÀ GIẢI NÉN LFW CHUẨN (KHÔNG GÂY TRÀN RAM)
# ------------------------------------------------------------------------------
print("\n--> [2/4] Chuẩn bị dữ liệu LFW chuẩn 6.000 cặp (Table 9)...")

# Dùng helper của sklearn để tải và giải nén LFW-Funneled an toàn
from sklearn.datasets._lfw import _check_fetch_lfw
if os.environ.get('STYLEID_LFW_HOME'):
    lfw_home = os.path.abspath(os.environ['STYLEID_LFW_HOME'])
    data_folder_path = os.path.join(lfw_home, 'lfw_funneled')
else:
    lfw_home, data_folder_path = _check_fetch_lfw()
pairs_path = os.path.join(lfw_home, 'pairs.txt')

print(f"  + Thư mục ảnh LFW: {data_folder_path}")
print(f"  + File giao thức cặp: {pairs_path}")

# Đọc file pairs.txt chuẩn quốc tế (10 Folds x 600 cặp = 6.000 cặp)
with open(pairs_path, 'r') as f:
    lines = [line.strip().split() for line in f if line.strip()]

folds = int(lines[0][0])
pairs_per_fold = int(lines[0][1])

pairs_list = []
idx = 1
for fold in range(folds):
    # Cùng người (Same / Positive)
    for _ in range(pairs_per_fold):
        p = lines[idx]
        name, num1, num2 = p[0], int(p[1]), int(p[2])
        p1 = os.path.join(data_folder_path, name, f"{name}_{num1:04d}.jpg")
        p2 = os.path.join(data_folder_path, name, f"{name}_{num2:04d}.jpg")
        pairs_list.append((p1, p2, 1, fold))
        idx += 1
    # Khác người (Different / Negative)
    for _ in range(pairs_per_fold):
        p = lines[idx]
        name1, num1, name2, num2 = p[0], int(p[1]), p[2], int(p[3])
        p1 = os.path.join(data_folder_path, name1, f"{name1}_{num1:04d}.jpg")
        p2 = os.path.join(data_folder_path, name2, f"{name2}_{num2:04d}.jpg")
        pairs_list.append((p1, p2, 0, fold))
        idx += 1

if folds != 10 or pairs_per_fold != 300 or len(pairs_list) != 6000:
    raise ValueError('This entry point requires LFW View2: 10 folds / 6000 pairs, not dev-test1000')
_missing = [p for pair in pairs_list for p in pair[:2] if not os.path.isfile(p)]
if _missing:
    raise FileNotFoundError(f'Missing LFW images; first missing: {_missing[0]}')
y_true = np.array([p[2] for p in pairs_list])
folds_arr = np.array([p[3] for p in pairs_list])

unique_image_paths = sorted(list(set([p[0] for p in pairs_list] + [p[1] for p in pairs_list])))
print(f"  + ✅ Đã nạp: {len(pairs_list)} cặp test ({np.sum(y_true==1)} Cùng người / {np.sum(y_true==0)} Khác người)")
print(f"  + Tổng số ảnh thực tế cần trích xuất: {len(unique_image_paths)} ảnh vuông 250x250 (Tiết kiệm 50% thời gian!)")

# ------------------------------------------------------------------------------
# 3. TRÍCH XUẤT ĐẶC TRƯNG SIÊU NHANH BẰNG CACHE EMBEDDING (~15 GIÂY)
# ------------------------------------------------------------------------------
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

def extract_all_embeddings(model, head=None, is_official=True, batch_size=128):
    model.eval()
    if head is not None:
        head.eval()
    
    emb_dict = {}
    with torch.no_grad():
        for i in range(0, len(unique_image_paths), batch_size):
            batch_paths = unique_image_paths[i:i+batch_size]
            # Mở ảnh bằng PIL - giữ nguyên tỉ lệ 1:1 vuông, chống méo mặt
            batch_imgs = [Image.open(p).convert('RGB') for p in batch_paths]
            
            inputs = processor(images=batch_imgs, return_tensors='pt')['pixel_values'].to(device)
            feats = model.visual_projection(model.vision_model(pixel_values=inputs).pooler_output)
            
            if is_official:
                embs = F.normalize(feats, dim=-1)
            else:
                embs = head(feats)
                
            embs_np = embs.cpu().numpy()
            for path, emb in zip(batch_paths, embs_np):
                emb_dict[path] = emb
                
    # Tính Cosine Similarity cho toàn bộ 6.000 cặp
    sim_scores = []
    for p1, p2, _, _ in pairs_list:
        e1 = emb_dict[p1]
        e2 = emb_dict[p2]
        sim_scores.append(float(np.dot(e1, e2)))
        
    return np.array(sim_scores)

# ------------------------------------------------------------------------------
# 4. HÀM TÍNH TOÁN METRICS TOÀN DIỆN (TABLE 9 & 10-FOLD CV)
# ------------------------------------------------------------------------------
def evaluate_metrics(y, scores, fold_indices):
    auc = roc_auc_score(y, scores)
    
    fpr, tpr, _ = roc_curve(y, scores)
    idx_1fpr = np.where(fpr <= 0.01)[0]
    tpr_1fpr = tpr[idx_1fpr[-1]] if len(idx_1fpr) > 0 else 0.0
    
    # Acc@0.3
    acc_03 = np.mean((scores >= 0.30).astype(int) == y)
    
    # Best Acc & Ngưỡng tối ưu
    best_acc, best_th = 0.0, 0.0
    for th in np.linspace(-0.2, 0.95, 250):
        acc = np.mean((scores >= th).astype(int) == y)
        if acc > best_acc:
            best_acc, best_th = acc, th
            
    pos_sim = np.mean(scores[y == 1])
    neg_sim = np.mean(scores[y == 0])
    
    # 10-Fold Cross Validation chuẩn
    fold_accs = []
    for fold in range(10):
        val_mask = (fold_indices == fold)
        train_mask = ~val_mask
        
        # Tìm ngưỡng tối ưu trên 9 fold
        best_f_acc, best_f_th = 0.0, 0.0
        for th in np.linspace(-0.2, 0.95, 200):
            acc = np.mean((scores[train_mask] >= th).astype(int) == y[train_mask])
            if acc > best_f_acc:
                best_f_acc, best_f_th = acc, th
                
        # Đánh giá trên fold còn lại
        fold_acc = np.mean((scores[val_mask] >= best_f_th).astype(int) == y[val_mask])
        fold_accs.append(fold_acc)
        
    return {
        'tpr_1fpr': tpr_1fpr,
        'acc_03': acc_03,
        'auc': auc,
        'best_acc': best_acc,
        'best_th': best_th,
        'pos_sim': pos_sim,
        'neg_sim': neg_sim,
        'margin': pos_sim - neg_sim,
        '10fold_acc': np.mean(fold_accs),
        '10fold_std': np.std(fold_accs)
    }

# ------------------------------------------------------------------------------
# 5. CHẤM ĐIỂM STYLEID CHÍNH CHỦ (kwanY/styleid)
# ------------------------------------------------------------------------------
print("\n--> [3/4] Đang nạp mô hình StyleID Chính chủ từ Hugging Face (kwanY/styleid)...")
styleid_model = CLIPModel.from_pretrained("kwanY/styleid").to(device)

print("  + Đang trích xuất đặc trưng 7.701 ảnh LFW...")
scores_styleid = extract_all_embeddings(styleid_model, is_official=True, batch_size=128)
metrics_styleid = evaluate_metrics(y_true, scores_styleid, folds_arr)
del styleid_model
torch.cuda.empty_cache()

# ------------------------------------------------------------------------------
# 6. CHẤM ĐIỂM DREAM SOUP (NẾU CÓ TRÊN DRIVE HOẶC CÙNG THƯ MỤC)
# ------------------------------------------------------------------------------
soup_ckpt_lora = None
for d in [str(CHECKPOINT_ROOT), '/content/drive/MyDrive/StyleID_A100_Champion', '/content/drive/MyDrive', '.']:
    p_l = os.path.join(d, 'disentangled_lora_clip_SOUP_S6_A100.pt')
    p_h = os.path.join(d, 'disentangled_head_SOUP_S6_A100.pt')
    if os.path.exists(p_l) and os.path.exists(p_h):
        soup_ckpt_lora, soup_ckpt_head = p_l, p_h
        break

metrics_soup = None
if not soup_ckpt_lora:
    raise FileNotFoundError('Missing exact Soup backbone/head. Set STYLEID_CHECKPOINT_DIR.')
if soup_ckpt_lora:
    print(f"\n--> [4/4] Phát hiện DREAM SOUP 🍲 tại Drive! Đang đối đầu...")
    class LoRALinear(nn.Module):
        def __init__(self, linear_layer, r=8, alpha=1.0):
            super().__init__()
            self.linear, self.scaling = linear_layer, alpha / r
            self.lora_down = nn.Linear(linear_layer.in_features, r, bias=False)
            self.lora_up = nn.Linear(r, linear_layer.out_features, bias=False)
        def forward(self, x):
            return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scaling

    class DisentangledHead(nn.Module):
        def __init__(self, clip_dim=768, id_dim=512):
            super().__init__()
            self.id_proj = nn.Sequential(nn.Linear(clip_dim, id_dim), nn.BatchNorm1d(id_dim), nn.PReLU())
        def forward(self, feat):
            return F.normalize(self.id_proj(feat), dim=-1)

    soup_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    for p in soup_model.parameters(): p.requires_grad = False
    for layer in soup_model.vision_model.encoder.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=8).to(device)
        layer.self_attn.k_proj = LoRALinear(layer.self_attn.k_proj, r=8).to(device)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=8).to(device)
        layer.self_attn.out_proj = LoRALinear(layer.self_attn.out_proj, r=8).to(device)
        
    soup_head = DisentangledHead().to(device)
    soup_model.load_state_dict(torch.load(soup_ckpt_lora, map_location=device), strict=False)
    soup_head.load_state_dict(torch.load(soup_ckpt_head, map_location=device), strict=False)
    
    scores_soup = extract_all_embeddings(soup_model, soup_head, is_official=False, batch_size=128)
    metrics_soup = evaluate_metrics(y_true, scores_soup, folds_arr)
    del soup_model, soup_head
    torch.cuda.empty_cache()

# ------------------------------------------------------------------------------
# 7. IN BẢNG ĐỐI CHIẾU CHUẨN BÀI BÁO TABLE 9
# ------------------------------------------------------------------------------
print("\n" + "="*96)
print("BẢNG ĐỐI CHIẾU KẾT QUẢ LFW THỰC NGHIỆM VỚI BÀI BÁO CÔNG BỐ (TABLE 9 - 6.000 CẶP)")
print("="*96)
print(f"{'Phương pháp':<32} | {'TPR↑ (@1% FPR)':<16} | {'Acc@0.3↑':<10} | {'AUC↑':<10} | {'10-Fold Acc↑':<14}")
print("-" * 96)
print(f"{'ArcFace (Bài báo công bố)':<32} | 0.9970           | 0.9975     | 0.9989     | ~99.8%")
print(f"{'StyleID (Bài báo công bố)':<32} | 0.9526           | 0.9750     | 0.9967     | —")
print("-" * 96)
print(f"{'StyleID (Tái lập mã nguồn này)':<32} | {metrics_styleid['tpr_1fpr']:<16.4f} | {metrics_styleid['acc_03']:<10.4f} | {metrics_styleid['auc']:<10.4f} | {metrics_styleid['10fold_acc']*100:.2f} ± {metrics_styleid['10fold_std']*100:.2f}%")

if metrics_soup:
    print(f"{'Disentangled (DREAM SOUP 🍲)':<32} | {metrics_soup['tpr_1fpr']:<16.4f} | {metrics_soup['acc_03']:<10.4f} | {metrics_soup['auc']:<10.4f} | {metrics_soup['10fold_acc']*100:.2f} ± {metrics_soup['10fold_std']*100:.2f}%")

print("=" * 96)
print("\nCHI TIẾT ĐỘ TƯƠNG ĐỒNG KHÔNG GIAN BIỂU DIỄN:")
print(f"  + StyleID Tái lập : Cùng người = {metrics_styleid['pos_sim']:.4f} | Khác người = {metrics_styleid['neg_sim']:.4f} | Margin = {metrics_styleid['margin']:.4f}")
if metrics_soup:
    print(f"  + Dream Soup 🍲   : Cùng người = {metrics_soup['pos_sim']:.4f} | Khác người = {metrics_soup['neg_sim']:.4f} | Margin = {metrics_soup['margin']:.4f}")
    print(f"  + Best Acc Soup   : {metrics_soup['best_acc']*100:.2f}% (tại ngưỡng tối ưu th* = {metrics_soup['best_th']:.2f})")
print("=" * 96)
