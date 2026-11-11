# Kết quả benchmark StyleID

[Bài báo StyleID](https://arxiv.org/abs/2604.21689v1):
Để chạy từ ảnh và checkpoint gốc, mở [notebook benchmark](colab/Kiem_Chung_Full_4Models_Colab.ipynb) trên Colab, chọn GPU rồi bấm **Run all**. Notebook cho kết quả thực nghiệm chạy lại; các số StyleID trích từ bài báo trong tài liệu này là số tham chiếu công bố, không phải đầu ra của notebook.

TPR@1% FPR là tỷ lệ chấp nhận đúng cặp cùng người khi tỷ lệ chấp nhận nhầm tối đa 1%. Acc@t là độ chính xác tại ngưỡng cosine t. AUROC đo khả năng xếp cặp cùng người cao hơn cặp khác người. Ô in đậm đánh dấu giá trị cao nhất của từng cột trước khi làm tròn.

## Bảng 1. Xác minh SKSF-A (1.876 cặp)

| Methods                    |   TPR@1% FPR ↑ |   Acc@0.3 ↑ |   Acc@0.4 ↑ |   Acc@0.5 ↑ |    AUROC ↑ |
|----------------------------|----------------|-------------|-------------|-------------|------------|
| StyleID                    |         0.8891 |      0.8731 |      0.7393 |      0.6178 |     0.9922 |
| ID + Style không trực giao |         0.7783 |      0.9392 |      0.9339 |      0.8625 |     0.9877 |
| Disentangle ID512          |         0.7825 |      0.9041 |  **0.9606** |  **0.9211** |     0.9914 |
| Dual-Branch 1280D          |     **0.9424** |  **0.9728** |      0.9104 |      0.7564 | **0.9968** |



## Bảng 2. Xác minh LFW View 2 (6.000 cặp)

Trên LFW, Dual-Branch 1280D kết hợp điểm cosine của StyleID 768D và ID512 sau adapter Hard-Positive theo tỷ lệ 50/50.

| Methods                    |   TPR@1% FPR ↑ |   Acc@0.3 ↑ |   Acc@0.4 ↑ |   Acc@0.5 ↑ |    AUROC ↑ |
|----------------------------|----------------|-------------|-------------|-------------|------------|
| StyleID                    |         0.9526 |      0.9750 |     0.9470 |     0.8782 |     0.9967 |
| ID + Style không trực giao |         0.9733 |      0.8588 |      0.9475 |      0.9813 |     0.9984 |
| Disentangle ID512          |         0.9860 |      0.8207 |      0.9358 |  **0.9817** |     0.9992 |
| Dual-Branch 1280D          |     **0.9877** |  **0.9762** |  **0.9892** |      0.9660 | **0.9993** |

Nguồn dòng StyleID: TPR, Acc@0.3 và AUROC lấy từ Table 9, trang 11 của bài báo. † Acc@0.4 và Acc@0.5 dùng số chạy lại vì Table 9 không công bố hai chỉ số này.

### Độ chính xác 10-fold LFW

- StyleID — 97.37% ± 0.72%
- ID + Style không trực giao — 98.38% ± 0.54%
- Disentangle ID512 — 98.58% ± 0.34%
- Dual-Branch 1280D — 98.87% ± 0.31%

Mỗi fold chọn ngưỡng trên chín fold còn lại. Kết quả là trung bình ± độ lệch chuẩn của mười fold.  Kết quả 10-fold của StyleID dùng số chạy lại vì bài báo không công bố chỉ số này.

## Bảng 3. Truy vấn danh tính SKSF-A (134 ảnh gallery, 938 truy vấn)

| Methods                    |   Rank-1 ↑ |   Rank-5 ↑ |      MRR ↑ |
|----------------------------|------------|------------|------------|
| StyleID                    |    0.7804 |    0.9382 |    0.8542 |
| ID + Style không trực giao |     0.7559 |     0.9339 |     0.8347 |
| Disentangle ID512          |     0.7655 |     0.9531 |     0.8483 |
| Dual-Branch 1280D          | **0.8817** | **0.9829** | **0.9266** |

† Dòng StyleID dùng số chạy lại cho cả Rank-1, Rank-5 và MRR vì bài báo không công bố ba chỉ số này cho giao thức SKSF-A gồm 134 ảnh gallery và 938 truy vấn.
