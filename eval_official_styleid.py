import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import load_image
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'--> Đang sử dụng thiết bị: {device.upper()}')

DATA_ROOT = 'styleid-s'

class MultiStyleDataset(Dataset):
    def __init__(self, root, is_train=False, val_split=0.2, seed=42):
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
        print(f'[{split_name}] {len(self.samples)} ảnh | {len(self.class_to_idx)} IDs | Styles: {list(self.style_to_idx.keys())}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, id_lbl, sty_lbl, ident, style = self.samples[idx]
        img = load_image(path).convert('RGB')
        return img, id_lbl, sty_lbl

def evaluate_cross_style(val_embs, val_id_labels, val_style_labels):
    sim_matrix = (val_embs @ val_embs.T).cpu()

    in_style_pos = []
    cross_style_pos = []
    neg_sims = []

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

    if cross_style_pos:
        cross_acc_04 = sum([1 for s in cross_style_pos if s >= 0.4]) / len(cross_style_pos)
        mean_cross_pos = np.mean(cross_style_pos)
    else:
        cross_acc_04, mean_cross_pos = 0.0, 0.0

    mean_neg = np.mean(neg_sampled) if neg_sampled else 0.0
    overall_acc_04 = (sum([1 for s in all_pos if s >= 0.4]) + sum([1 for s in neg_sampled if s < 0.4])) / (len(all_pos) + len(neg_sampled))

    return mean_cross_pos, mean_neg, cross_acc_04, overall_acc_04, len(cross_style_pos), len(in_style_pos)

def get_features(m, pixel_values):
    vision_outputs = m.vision_model(pixel_values=pixel_values)
    return m.visual_projection(vision_outputs.pooler_output)

def test_official_styleid():
    print('\n--> Đang tải mô hình SOTA CHÍNH CHỦ từ Hugging Face: kwanY/styleid...')
    processor = CLIPProcessor.from_pretrained('kwanY/styleid')
    model = CLIPModel.from_pretrained('kwanY/styleid').to(device)
    model.eval()

    val_dataset = MultiStyleDataset(DATA_ROOT, is_train=False, val_split=0.2, seed=42)

    print('\n--> Đang trích xuất đặc trưng và đánh giá Cross-Style...')
    val_embs, val_id_labels, val_style_labels = [], [], []

    with torch.no_grad():
        for i in tqdm.tqdm(range(min(len(val_dataset), 500)), desc='Đánh giá mô hình chính chủ'):
            img, id_lbl, sty_lbl = val_dataset[i]
            inputs = processor(images=img, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(device)
            emb = get_features(model, pixel_values)
            emb = F.normalize(emb, dim=-1)
            val_embs.append(emb)
            val_id_labels.append(id_lbl)
            val_style_labels.append(sty_lbl)

    val_embs = torch.cat(val_embs, dim=0)
    mean_cross_pos, mean_neg, cross_acc, overall_acc, n_cross, n_in = evaluate_cross_style(
        val_embs, val_id_labels, val_style_labels
    )

    print(f'\n=======================================================')
    print(f' [KẾT QUẢ MÔ HÌNH CHÍNH CHỦ (kwanY/styleid - TRAIN FULL 135GB)]')
    print(f' - ĐỘ CHÍNH XÁC CROSS-STYLE (Khác phong cách):    {cross_acc * 100:.2f}%')
    print(f' - Độ chính xác tổng thể (Overall Accuracy@0.4):   {overall_acc * 100:.2f}%')
    print(f' - Độ tương đồng CÙNG NGƯỜI (Khác style):         {mean_cross_pos:.4f}')
    print(f' - Độ tương đồng KHÁC NGƯỜI:                       {mean_neg:.4f}')
    print(f' - Số cặp test: {n_cross} cặp Cross-Style / {n_in} cặp cùng style')
    print(f'=======================================================\n')

if __name__ == '__main__':
    test_official_styleid()
