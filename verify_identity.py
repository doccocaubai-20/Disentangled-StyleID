import sys
import argparse
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="StyleID: So sánh độ tương đồng danh tính khuôn mặt giữa 2 ảnh.")
    parser.add_argument("--img1", type=str, default="SKSF-A_data/Photo/1.png", help="Đường dẫn ảnh 1")
    parser.add_argument("--img2", type=str, default="SKSF-A_data/style1/1.png", help="Đường dẫn ảnh 2")
    parser.add_argument("--threshold", type=float, default=0.4, help="Ngưỡng cùng danh tính (mặc định 0.4 theo bài báo)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị thực thi: {device.upper()}")

    print("Đang nạp mô hình StyleID từ cache...")
    model = CLIPModel.from_pretrained("kwanY/styleid").to(device)
    processor = CLIPProcessor.from_pretrained("kwanY/styleid")

    def get_embedding(path):
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            return emb / emb.norm(dim=-1, keepdim=True)

    print(f"Đang phân tích: '{args.img1}' và '{args.img2}' ...")
    emb1 = get_embedding(args.img1)
    emb2 = get_embedding(args.img2)

    similarity = (emb1 @ emb2.T).item()

    print("\n" + "=" * 50)
    print(f"Độ tương đồng Cosine: {similarity:.4f}")
    print(f"Ngưỡng xác định cùng người (Threshold): {args.threshold}")
    if similarity >= args.threshold:
        print("=> KẾT LUẬN: CÙNG MỘT NGƯỜI (SAME IDENTITY) [MATCH] ✅")
    else:
        print("=> KẾT LUẬN: KHÁC NGƯỜI (DIFFERENT IDENTITIES) [NO MATCH] ❌")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
