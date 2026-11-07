# Kết quả kiểm chứng Colab

## SKSF A xác minh danh tính

1.876 cặp gồm 938 cặp cùng người và 938 cặp khác người, lấy từ 1.072 ảnh.

| Phương pháp | TPR@1% FPR ↑ | Acc@0.3 ↑ | Acc@0.4 ↑ | Acc@0.5 ↑ | AUROC ↑ |
|---|---:|---:|---:|---:|---:|
| StyleID chạy lại | 86.89% | 86.73% | 73.51% | 60.61% | 0.993052 |
| ID + Style không trực giao | 77.83% | 93.92% | 93.39% | 86.25% | 0.987676 |
| Disentangle ID512 | 78.25% | 90.41% | **96.06%** | **92.11%** | 0.991421 |
| Dual-Branch 1280D | **94.24%** | **97.28%** | 91.04% | 75.64% | **0.996774** |

## LFW xác minh danh tính

6.000 cặp View 2, 7.701 ảnh xuất hiện trong danh sách cặp, 10 fold.

| Phương pháp | TPR@1% FPR ↑ | Acc@0.3 ↑ | AUC ↑ | 10-fold Acc ↑ |
|---|---:|---:|---:|---:|
| StyleID chạy lại | 94.33% | 97.45% | 0.997133 | 97.37% ± 0.72% |
| ID + Style không trực giao | 97.33% | 85.88% | 0.998415 | 98.38% ± 0.54% |
| Disentangle ID512 | 98.60% | 82.07% | 0.999174 | 98.58% ± 0.34% |
| Dual-Branch 1280D | **98.80%** | 96.10% | **0.999283** | **98.87% ± 0.38%** |
| Dual-Branch + Hard-Positive | 98.77% | **97.62%** | 0.999262 | **98.87% ± 0.31%** |

## SKSF A truy vấn danh tính

134 ảnh chụp làm gallery, 938 ảnh phác thảo làm truy vấn.

| Phương pháp | Rank-1 ↑ | Rank-5 ↑ | MRR ↑ |
|---|---:|---:|---:|
| StyleID chạy lại | 78.04% | 93.82% | 0.8542 |
| ID + Style không trực giao | 75.59% | 93.39% | 0.8347 |
| Disentangle ID512 | 76.55% | 95.31% | 0.8483 |
| Dual-Branch 1280D | **88.17%** | **98.29%** | **0.9266** |

## Cách đọc kết quả

Baseline là StyleID chạy lại. Dual-Branch dùng cả StyleID 768D và Disentangle ID512 với trọng số cosine 50/50. Hard-Positive là biến thể riêng ở bảng LFW. Các giá trị là kết quả lượt Colab đã nhận.

TPR@1% FPR là tỷ lệ chấp nhận đúng cặp cùng người với tỷ lệ chấp nhận nhầm tối đa 1%. Acc@t là tỷ lệ phân loại đúng với ngưỡng cosine t. AUC và AUROC đo khả năng xếp cặp cùng người cao hơn cặp khác người. 10-fold Acc ghi trung bình và độ lệch chuẩn, ngưỡng được chọn trên 9 fold và chấm ở fold còn lại.

Rank-1 và Rank-5 là tỷ lệ tìm đúng trong một hoặc năm vị trí đầu. MRR là trung bình nghịch đảo thứ hạng kết quả đúng.

Dual-Branch cao hơn StyleID chạy lại ở cả năm cột SKSF-A và cả ba chỉ số truy vấn. Trên LFW, Dual-Branch có TPR và AUC cao nhất trong các dòng, còn biến thể Hard-Positive có Acc@0.3 cao nhất. Các cấu hình có lịch train khác nhau, nên bảng chưa cô lập tác động của riêng loss trực giao.

[Mở notebook kiểm chứng](../03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.ipynb) · [Bảng Word có thể chỉnh sửa](KET_QUA_KIEM_CHUNG_4_MODELS_COLAB.docx)
