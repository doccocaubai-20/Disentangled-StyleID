# Disentangled StyleID

Mã nguồn cuối cùng để kiểm chứng StyleID, ID + Style không trực giao, Disentangle ID512 và Dual-Branch 1280 chiều trên SKSF-A và LFW.

## Chạy ba bảng trên Colab

[Mở notebook kiểm chứng](https://colab.research.google.com/github/doccocaubai-20/Disentangled-StyleID/blob/main/StyleID_Core/03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.ipynb), chọn GPU và bấm **Run all**. Notebook tự cài thư viện, tải ảnh và checkpoint, rồi tính ba bảng từ ảnh gốc. Các file Drive cần có quyền tải và còn hạn mức.

- [Notebook kiểm chứng](StyleID_Core/03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.ipynb)
- [Mã Python tương ứng](StyleID_Core/03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.py)
- [Kết quả Colab cuối cùng](StyleID_Core/00_docs/KET_QUA_COLAB.md)
- [Bảng Word có thể chỉnh sửa](StyleID_Core/00_docs/KET_QUA_KIEM_CHUNG_4_MODELS_COLAB.docx)

## Mã chính

| Thư mục | Nội dung |
|---|---|
| [01_train](StyleID_Core/01_train/) | Recipe train có trực giao và đối chứng không trực giao |
| [02_soup](StyleID_Core/02_soup/) | Tạo trọng số Soup từ S6 và A100 |
| [03_benchmark/colab](StyleID_Core/03_benchmark/colab/) | Notebook và Python kiểm chứng cuối cùng |
| [04_demo](StyleID_Core/04_demo/) | Demo web Soup và Soup + LEACE |
| [00_docs](StyleID_Core/00_docs/) | Kết quả, vị trí trọng số và thông tin StyleID gốc |

[Hướng dẫn sử dụng](StyleID_Core/README_VI.md) · [Trọng số trên Drive](StyleID_Core/00_docs/TRAIN_VA_TRONG_SO_DRIVE.md)

## Cách đọc kết quả

Dual-Branch ghép vector StyleID 768 chiều với vector ID512 của Disentangle, với trọng số cosine 50/50. Biến thể Hard-Positive được ghi thành dòng riêng trong bảng LFW. Demo web hiện dùng Soup/LEACE và có hướng dẫn riêng trong Core.

Baseline là StyleID chạy lại trong cùng notebook. Các checkpoint có lịch train khác nhau, nên chênh lệch giữa các dòng chưa cô lập riêng tác động của loss trực giao. Giao thức cặp SKSF-A của nhóm đã được khóa. Tính năng truy vấn danh tính dùng 134 ảnh chụp làm gallery và 938 ảnh phác thảo làm truy vấn.

Thông tin trích dẫn và điều kiện sử dụng StyleID được lưu tại [tài liệu gốc](StyleID_Core/00_docs/STYLEID_ORIGINAL_README.md).
