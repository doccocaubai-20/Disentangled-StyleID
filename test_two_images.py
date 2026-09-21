# -*- coding: utf-8 -*-
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def load_image_clean(path):
    img = Image.open(path)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        canvas = Image.new('RGB', img.size, (255, 255, 255))
        canvas.paste(img, mask=img.split()[3])
        return canvas
    return img.convert('RGB')

def main():
    parser = argparse.ArgumentParser(description='So sánh 2 ảnh giữa Disentangled-StyleID và StyleID chính chủ.')
    parser.add_argument('--img1', type=str, default='data/1.jpg', help='Đường dẫn ảnh 1 (vd: ảnh chân dung thật)')
    parser.add_argument('--img2', type=str, default='data/2.jpg', help='Đường dẫn ảnh 2 (vd: tranh vẽ/sketch)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Ngưỡng cùng danh tính (mặc định 0.4)')
    args = parser.parse_args()

    if not os.path.exists(args.img1):
        print(f'❌ Không tìm thấy ảnh 1: {args.img1}')
        return
    if not os.path.exists(args.img2):
        print(f'❌ Không tìm thấy ảnh 2: {args.img2}')
        return

    print('=' * 75)
    print(f'🔬 SO TÀI NHẬN DIỆN KHUÔN MẶT BẤT BIẾN PHONG CÁCH GIỮA 2 ẢNH')
    print(f'   • Thiết bị thực thi: {device.upper()}')
    print(f'   • Ảnh 1: {args.img1}')
    print(f'   • Ảnh 2: {args.img2}')
    print(f'   • Ngưỡng chuẩn (Threshold): {args.threshold}')
    print('=' * 75)

    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

    print('\n[1/2] Đang nạp mô hình Disentangled-StyleID (Trọng số của bạn)...')
    my_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    for p in my_model.parameters(): p.requires_grad = False
    apply_lora_to_clip(my_model, r=8)
    
    lora_path = 'disentangled_lora_clip.pt'
    if os.path.exists(lora_path):
        my_model.load_state_dict(torch.load(lora_path, map_location=device), strict=False)
        print(f'   --> ✅ Đã nạp thành công LoRA: {lora_path}')
    else:
        print(f'   --> ⚠️ Không tìm thấy {lora_path}')

    my_head = DisentangledHead().to(device)
    head_path = 'disentangled_head.pt'
    if os.path.exists(head_path):
        my_head.load_state_dict(torch.load(head_path, map_location=device), strict=False)
        print(f'   --> ✅ Đã nạp thành công DisentangledHead: {head_path}')

    my_model.eval()
    my_head.eval()

    print('\n[2/2] Đang nạp mô hình gốc StyleID (kwanY/styleid - SIGGRAPH 2026)...')
    official_model = CLIPModel.from_pretrained('kwanY/styleid').to(device)
    official_model.eval()
    print('   --> ✅ Đã nạp thành công StyleID chính chủ.')

    im1 = load_image_clean(args.img1)
    im2 = load_image_clean(args.img2)

    pv1 = processor(images=im1, return_tensors='pt')['pixel_values'].to(device)
    pv2 = processor(images=im2, return_tensors='pt')['pixel_values'].to(device)

    with torch.no_grad():
        f1_my = get_features(my_model, pv1)
        f2_my = get_features(my_model, pv2)
        zid1 = my_head(f1_my)
        zid2 = my_head(f2_my)
        sim_my = (zid1 * zid2).sum(dim=-1).item()

        f1_off = official_model.get_image_features(pixel_values=pv1)
        f2_off = official_model.get_image_features(pixel_values=pv2)
        f1_off = F.normalize(f1_off, dim=-1)
        f2_off = F.normalize(f2_off, dim=-1)
        sim_off = (f1_off * f2_off).sum(dim=-1).item()

    print('\n' + '=' * 75)
    print('{:<30} | {:<15} | {:<20}'.format('MÔ HÌNH', 'ĐỘ TƯƠNG ĐỒNG', f'KẾT LUẬN TẠI NGƯỠNG {args.threshold}'))
    print('-' * 75)

    res_off = '✅ CÙNG MỘT NGƯỜI' if sim_off >= args.threshold else '❌ BỊ NHẬN NHẦM (KHÁC NGƯỜI)'
    print('{:<30} | {:<15.4f} | {}'.format('StyleID (Tác giả bài báo)', sim_off, res_off))

    res_my = '✅ CÙNG MỘT NGƯỜI' if sim_my >= args.threshold else '❌ BỊ NHẬN NHẦM (KHÁC NGƯỜI)'
    print('{:<30} | {:<15.4f} | {}'.format('Disentangled (Bản của bạn)', sim_my, res_my))
    print('=' * 75)

    diff = sim_my - sim_off
    if diff > 0:
        print(f'🏆 Bản của bạn TĂNG ĐỘ TỰ TIN CÙNG NGƯỜI LÊN: +{diff:.4f} (+{diff/max(abs(sim_off), 1e-4)*100:.1f}%)')
    else:
        print(f'Biên độ chênh lệch: {diff:.4f}')
    print('=' * 75 + '\n')

if __name__ == '__main__':
    main()
