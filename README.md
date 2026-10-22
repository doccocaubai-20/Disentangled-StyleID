# Disentangled-StyleID

Dự án phát triển từ StyleID: hai nhánh danh tính và phong cách, model soup và bộ lọc LEACE sau vector danh tính.

## Bắt đầu từ StyleID_Core

Mã hiện dùng và hướng dẫn: **[StyleID_Core](StyleID_Core/README_VI.md)**.

| Thư mục | Nội dung |
|---|---|
| `StyleID_Core/00_docs/` | Hướng dẫn và tài liệu tham chiếu |
| `StyleID_Core/01_train/` | Train S6/A100 và bản lịch sử |
| `StyleID_Core/02_soup/` | Kết hợp trọng số S6 + A100 |
| `StyleID_Core/03_benchmark/` | Test SKSF-A, LFW 6.000 cặp và Colab |
| `StyleID_Core/04_demo/` | Web so sánh StyleID, Soup và Soup + LEACE |
| `StyleID_Core/assets/` | Bộ lọc LEACE và metadata |

Trên Windows:

```text
cd StyleID_Core
python -m pip install -r requirements.txt
copy .env.example .env
```

Điền đường dẫn checkpoint trong `.env`, rồi bấm `mo_server.bat`. Trọng số lớn và dataset được cung cấp riêng. Ngưỡng trên web chỉ phục vụ demo, chưa hiệu chuẩn riêng cho từng model/chế độ.

## Mã tham chiếu giữ lại

- `full_train_clip-checkpoint.py`: mã train StyleID ban đầu, khác recipe hai nhánh của nhóm.
- `generation/`: mã tham chiếu tạo dữ liệu StyleBench; không phải pipeline tạo ảnh đã kiểm chứng với model của nhóm.
- `data/`: ảnh mẫu có sẵn trong repo.
- [README StyleID ban đầu](StyleID_Core/00_docs/STYLEID_ORIGINAL_README.md): mô tả, trích dẫn và thông báo sử dụng từ bản trước.

Các bản train/test trùng chức năng, báo cáo cũ, handover nội bộ và file thử đã được bỏ khỏi phiên bản hiện tại trên GitHub. Có thể truy vết trong lịch sử Git. Các file nghiên cứu còn trong workspace cá nhân không tự động được xuất bản lên repo.

## Phạm vi kết luận

Hai nhánh không bảo đảm tách hoàn toàn danh tính và phong cách. LEACE là phương pháp có sẵn được tích hợp và cần đánh giá cùng mức đánh đổi về nhận dạng. Các checkpoint có lịch train khác nhau không phải đối chứng cô lập tác động từng loss.
