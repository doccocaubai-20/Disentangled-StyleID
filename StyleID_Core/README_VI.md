# StyleID Core

Bộ mã gồm huấn luyện, tạo Soup, kiểm chứng cuối cùng và demo web. Checkpoint được tải từ Drive và Hugging Face. Các file huấn luyện chạy bằng Python.

## Kiểm chứng kết quả cuối cùng

Mở [Kiem_Chung_Full_4Models_Colab.ipynb](03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.ipynb) trên Colab, chọn GPU rồi bấm **Run all**.

Notebook tính lại từ ảnh và checkpoint cho bốn cấu hình StyleID, ID + Style không trực giao, Disentangle ID512 và Dual-Branch 1280D. Trên LFW, Dual-Branch 1280D dùng adapter Hard-Positive cho nhánh ID512. Hai nhánh được xử lý lần lượt để hạn chế dùng VRAM.

SKSF-A gồm 1.072 ảnh và 1.876 cặp. LFW dùng 7.701 ảnh xuất hiện trong 6.000 cặp View 2, chia 10 fold. Notebook kiểm tra SHA256 của checkpoint và ảnh SKSF-A, kiểm tra archive LFW và dừng khi thiếu ảnh. Drive, Hugging Face và Figshare cần truy cập được.

Hai bảng xác minh dùng chung các cột TPR@1% FPR, Acc@0.3, Acc@0.4, Acc@0.5 và AUROC. Điểm hiển thị với bốn chữ số thập phân và điểm cao nhất ở mỗi cột được in đậm. Notebook cũng hiển thị kết quả 10-fold của LFW dưới bảng LFW và bảng truy vấn danh tính của SKSF-A.

Xem [kết quả benchmark theo mẫu mới](03_benchmark/result_benchmark.md), [kết quả Colab đã lưu](00_docs/KET_QUA_COLAB.md) và [bản Word](00_docs/KET_QUA_KIEM_CHUNG_4_MODELS_COLAB.docx). Notebook tải adapter `.pt` từ Drive ở bước LFW.

## Các file cần dùng

| File | Công dụng |
|---|---|
| [01_train/train_no_ortho.py](01_train/train_no_ortho.py) | Train ID512 + style256 với loss trực giao bằng 0 |
| [01_train/train_a100_50.py](01_train/train_a100_50.py) | Recipe có trực giao, gộp shard [1,2,4,5,6] |
| [01_train/train_shard_0_den_6.py](01_train/train_shard_0_den_6.py) | Train nối tiếp, mỗi lần chọn một shard từ 0 đến 6 |
| [02_soup/model_soup_averaging.py](02_soup/model_soup_averaging.py) | Trộn hai cặp checkpoint S6 và A100 trên CPU |
| [04_demo/server.py](04_demo/server.py) | So sánh StyleID với Dual-Branch, Hard-Positive trên LFW và Disentangle ID512 |
| [MANIFEST.json](MANIFEST.json) | Danh sách và SHA256 các file phát hành |

## Huấn luyện và tạo Soup

Chạy từ thư mục `StyleID_Core` với Python 3.10+ và bản PyTorch phù hợp với thiết bị.

```text
python -m pip install torch numpy transformers==4.52.0 huggingface-hub pillow tqdm scikit-learn
python 01_train/train_no_ortho.py --help
python 01_train/train_a100_50.py --help
python 01_train/train_shard_0_den_6.py --help
python 02_soup/model_soup_averaging.py --help
```

Recipe và vị trí checkpoint được ghi trong [hướng dẫn huấn luyện và trọng số](00_docs/TRAIN_VA_TRONG_SO_DRIVE.md). Trainer tuần tự yêu cầu cặp backbone và head của shard trước trong `--work-dir` khi shard lớn hơn 0. Tạo Soup cần hai cặp checkpoint cha S6 và A100.

## Chạy demo web

```text
cd StyleID_Core
python -m pip install -r 04_demo/requirements.txt
python 04_demo/server.py
```

Mở [localhost:8000](http://localhost:8000) và bấm một cặp LFW. Demo hiển thị hai ảnh, nhãn của benchmark, điểm từng nhánh và kết quả của StyleID gốc với Dual-Branch + Hard-Positive tại ngưỡng mặc định 0,30.

| Cặp LFW View 2 | Kịch bản |
|---|---|
| `lfw-1342` | Cùng người, một ảnh đeo kính râm |
| `lfw-0557` | Hai người khác nhau, cặp âm khó |
| `lfw-0000` | Cặp dương đối chứng |

Lần đầu mở một cặp mẫu, server tải archive LFW funneled, kiểm tra SHA256 và giữ sáu ảnh mẫu trong `data/lfw-demo/`. Có thể trỏ đến archive đã tải bằng `STYLEID_LFW_ARCHIVE`. [Nguồn ảnh LFW funneled](https://ndownloader.figshare.com/files/5976015)

Model được nạp khi bắt đầu so sánh. StyleID và CLIP dùng cache Hugging Face. Demo tải hai file Soup từ Drive khi thiếu bản cục bộ. File adapter chỉ được tải khi so sánh một cặp mẫu LFW ở chế độ Hard-Positive. Với ảnh tự tải lên, có thể chọn Dual-Branch hoặc Disentangle ID512. Xem [mẫu cấu hình](04_demo/.env.example) để dùng checkpoint cục bộ hoặc đổi vị trí cache.

API `POST /api/compare` nhận `preset_id` cho cặp LFW hoặc `img1_base64` và `img2_base64` cho ảnh tải lên. Các chế độ gồm `dual`, `raw` và `dual_hp` cho cặp LFW. Thay đổi ngưỡng dùng lại điểm đã tính.
