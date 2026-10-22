# StyleID Core

Mã nguồn được chia theo chức năng. Bắt đầu từ thư mục gốc này; file `.env` và `mo_server.bat` đặt tại đây.

```text
StyleID_Core/
├── 00_docs/                   Hướng dẫn chi tiết
├── 01_train/                  Huấn luyện hai nhánh lịch sử
│   ├── train_a100_50.py
│   ├── train_shard_0_den_6.py
│   └── archive/               Các phiên bản train cũ để tham khảo
├── 02_soup/
│   └── tao_soup.py            Kết hợp trọng số S6 + A100
├── 03_benchmark/
│   ├── test_sksf_a.py         Kiểm thử SKSF-A
│   ├── test_lfw_full_6000.py  Kiểm thử LFW
│   └── colab/                Các file upload chạy trên Colab
├── 04_demo/
│   ├── server.py             Web so sánh StyleID, Soup và Soup + LEACE
│   └── demo_leace.py         Nạp và áp dụng bộ lọc
├── assets/                   Bộ lọc LEACE và metadata
├── checkpoints/soup/         Tự thêm hai file trọng số; không đưa lên Git
├── .env.example              Cấu hình mẫu
├── mo_server.bat             Mở web trên Windows
├── requirements.txt
└── MANIFEST.json             Nguồn và hash các file
```

## Mở demo

1. Cài Python và thư viện: `python -m pip install -r requirements.txt`.
2. Đặt hai trọng số Soup đúng tên vào `checkpoints/soup/`, hoặc điền `STYLEID_CHECKPOINT_DIR` trong `.env`.
3. Sao chép `.env.example` thành `.env` nếu cần tùy chỉnh đường dẫn. Dùng đường dẫn tuyệt đối. Biến môi trường hệ thống được ưu tiên hơn `.env`.
4. Bấm đúp `mo_server.bat`, hoặc chạy `python 04_demo/server.py`.

Tên hai trọng số bắt buộc:

```text
disentangled_lora_clip_SOUP_S6_A100.pt
disentangled_head_SOUP_S6_A100.pt
```

Server tự tìm bộ lọc trong `assets/`, kiểm tra hash bộ lọc và đúng checkpoint Soup. Lần đầu có thể tải CLIP và StyleID chính thức. Ảnh preset cần tài nguyên riêng; bạn có thể upload hai ảnh của mình trên web. Ngưỡng mặc định 0,4 chỉ để demo, chưa hiệu chuẩn riêng cho từng chế độ.

## Các lệnh khác

Chạy từ thư mục `StyleID_Core`:

```text
python 01_train/train_a100_50.py --help
python 01_train/train_shard_0_den_6.py --help
python 02_soup/tao_soup.py --help
python 03_benchmark/test_sksf_a.py
python 03_benchmark/test_lfw_full_6000.py
```

Các file trong `03_benchmark/colab/` là file độc lập để upload lên notebook Colab; không phải lệnh mở demo. Đọc [hướng dẫn chi tiết](00_docs/HUONG_DAN_CHI_TIET.md) trước khi train hoặc tạo Soup mới.

Các recipe train ở đây là bản lịch sử S6/A100, không phải file đối chứng full10shard vừa tạo trong `Ablation_Thay_4Dong` của workspace. Trọng số lớn, dataset, `.env` và cache Python không được commit lên GitHub.
