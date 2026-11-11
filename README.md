# Disentangled StyleID

Mã nguồn cuối cùng để kiểm chứng StyleID, ID + Style không trực giao, Disentangle ID512 và Dual-Branch 1280 chiều trên SKSF-A và LFW.

## Chạy ba bảng trên Colab

[Mở notebook kiểm chứng](https://colab.research.google.com/github/doccocaubai-20/Disentangled-StyleID/blob/main/StyleID_Core/03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.ipynb), chọn GPU và bấm **Run all**. Notebook tự cài thư viện, tải ảnh và checkpoint, rồi tính ba bảng từ ảnh gốc. Các file Drive cần có quyền tải và còn hạn mức.

- [Kết quả benchmark SKSF-A và LFW](StyleID_Core/03_benchmark/result_benchmark.md)
- [Bảng Word có thể chỉnh sửa](StyleID_Core/00_docs/KET_QUA_KIEM_CHUNG_4_MODELS_COLAB.docx)

## Mã chính

| Thư mục | Nội dung |
|---|---|
| [01_train](StyleID_Core/01_train/) | Các script Python huấn luyện, gồm [đối chứng không trực giao](StyleID_Core/01_train/train_no_ortho.py) |
| [02_soup](StyleID_Core/02_soup/) | Tạo trọng số Soup từ S6 và A100 |
| [03_benchmark/colab](StyleID_Core/03_benchmark/colab/) | Notebook kiểm chứng duy nhất cho SKSF-A và LFW |
| [04_demo](StyleID_Core/04_demo/) | So sánh StyleID gốc với Dual-Branch + Hard-Positive trên ba cặp LFW |
| [00_docs](StyleID_Core/00_docs/) | Kết quả, vị trí trọng số và thông tin StyleID gốc |

[Hướng dẫn sử dụng](StyleID_Core/README_VI.md) · [Trọng số trên Drive](StyleID_Core/00_docs/TRAIN_VA_TRONG_SO_DRIVE.md)

## Cách đọc kết quả

Hai bảng xác minh dùng cùng các cột TPR@1% FPR, Acc@0.3, Acc@0.4, Acc@0.5 và AUROC. Notebook hiển thị bốn chữ số thập phân và in đậm kết quả cao nhất của từng cột.

Dual-Branch ghép vector StyleID 768 chiều với vector Soup ID512, với trọng số cosine 50/50. Trên LFW, dòng Dual-Branch 1280D áp dụng adapter Hard-Positive cho nhánh Soup ID512 trước khi ghép. Các cặp mẫu LFW trong demo dùng cùng cấu hình với ngưỡng 0,30.

Baseline là StyleID chạy lại trong cùng notebook. Các checkpoint có lịch train khác nhau, nên chênh lệch giữa các dòng chưa cô lập riêng tác động của loss trực giao. Giao thức cặp SKSF-A của nhóm đã được khóa. Tính năng truy vấn danh tính dùng 134 ảnh chụp làm gallery và 938 ảnh phác thảo làm truy vấn.

Thông tin trích dẫn và điều kiện sử dụng StyleID được lưu tại [tài liệu gốc](StyleID_Core/00_docs/STYLEID_ORIGINAL_README.md).
