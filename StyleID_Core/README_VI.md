# StyleID Core

Bộ mã gồm huấn luyện, tạo Soup, kiểm chứng cuối cùng và demo web. Trọng số và dataset được cung cấp qua Drive hoặc nguồn tải trong notebook.

## Kiểm chứng kết quả cuối cùng

Mở [Kiem_Chung_Full_4Models_Colab.ipynb](03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.ipynb) trên Colab, chọn GPU rồi bấm **Run all**.

Notebook tính lại từ ảnh và checkpoint cho bốn cấu hình StyleID, ID + Style không trực giao, Disentangle ID512 và Dual-Branch 1280D. Bảng LFW có thêm biến thể Dual-Branch + Hard-Positive. Hai nhánh được xử lý lần lượt để hạn chế dùng VRAM.

SKSF-A gồm 1.072 ảnh và 1.876 cặp. LFW dùng 7.701 ảnh xuất hiện trong 6.000 cặp View 2, chia 10 fold. Notebook kiểm tra SHA256 của checkpoint và ảnh SKSF-A, kiểm tra archive LFW và dừng khi thiếu ảnh. Drive, Hugging Face và Figshare cần truy cập được.

Có thể upload [file Python tương ứng](03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.py) vào Colab và chạy bằng `runpy.run_path`. File này cài thư viện ở đầu và cần GPU. Trên máy Windows, ưu tiên dùng notebook Colab.

Kết quả đã nhận được lưu trong [bảng Markdown](00_docs/KET_QUA_COLAB.md) và [bản Word](00_docs/KET_QUA_KIEM_CHUNG_4_MODELS_COLAB.docx). Biến thể Hard-Positive dùng adapter đã nhúng trong notebook.

## Các file cần dùng

| File | Công dụng |
|---|---|
| `01_train/train_a100_50.py` | Recipe có trực giao, gộp shard [1,2,4,5,6] |
| `01_train/train_shard_0_den_6.py` | Trainer nối tiếp, mỗi lần chọn một shard 0 đến 6 |
| `01_train/colab/Dong2_Train_ID_Style_Khong_Ortho.ipynb` | Notebook train đối chứng, giữ loss ID và style, tắt loss trực giao |
| `01_train/colab/Dong2_Train_ID_Style_Khong_Ortho.py` | Mã Python của notebook đối chứng |
| `02_soup/tao_soup.py` | Trộn hai cặp checkpoint S6 và A100 trên CPU |
| `03_benchmark/colab/Kiem_Chung_Full_4Models_Colab.ipynb` | Chạy ba bảng kiểm chứng từ ảnh và checkpoint |
| `04_demo/server.py` | Web demo Dual-Branch + Hard-Positive, mặc định ngưỡng 0,30 |
| `04_demo/demo_hard_positive.py` và `assets/lfw_hard_positive_*` | Adapter và metadata dùng đúng trọng số nhúng trong notebook |
| `04_demo/demo_leace.py` | Bộ lọc cho chế độ LEACE |
| `assets/soup_leace_operator.npz` và metadata | Bộ lọc bắt buộc của demo LEACE |
| `.env.example` | Cấu hình đường dẫn cho demo |
| `MANIFEST.json` | Danh sách và SHA256 các file phát hành |

## Huấn luyện và tạo Soup

Đọc [recipe và vị trí checkpoint](00_docs/TRAIN_VA_TRONG_SO_DRIVE.md) trước khi chạy. Các recipe có trực giao là mã huấn luyện lịch sử. Shard 0 dùng một phần dữ liệu và các lượt đầu có thiết lập riêng, nên chạy mới trainer tuần tự chưa bảo đảm tái tạo đúng checkpoint S6 lịch sử.

Chạy từ thư mục `StyleID_Core` trên môi trường phù hợp.

```text
python 01_train/train_a100_50.py --help
python 01_train/train_shard_0_den_6.py --help
python 02_soup/tao_soup.py --help
```

Trainer tuần tự yêu cầu cả backbone và head của shard trước trong `--work-dir` khi shard lớn hơn 0. Tạo Soup cần đúng hai checkpoint cha S6 và A100. Đầu ra mới được đặt trong thư mục riêng, và chương trình kiểm tra trước khi ghi đè.

Mã lịch sử chi tiết hơn có thể truy vết trong lịch sử Git. Notebook đối chứng chạy hai epoch trên năm shard gộp chung [1,2,4,5,6], khởi tạo mới. Các dòng benchmark chưa phải đối chứng có lịch train hoàn toàn giống nhau.

## Demo web trên Windows

```text
cd StyleID_Core
python -m pip install -r requirements.txt
copy .env.example .env
```

Đặt hai checkpoint Soup vào `checkpoints/soup/` hoặc điền `STYLEID_CHECKPOINT_DIR` trong `.env`.

```text
disentangled_lora_clip_SOUP_S6_A100.pt
disentangled_head_SOUP_S6_A100.pt
```

Bấm `mo_server.bat` hoặc chạy `python 04_demo/server.py`. Giữ toàn bộ `04_demo/` và `assets/` cùng mã demo. Server kiểm tra hash checkpoint, bộ lọc và adapter. Lần đầu chạy có thể tải CLIP và StyleID.

Web mặc định đối chiếu StyleID với **Dual-Branch + Hard-Positive**, cùng cấu hình dòng cuối bảng LFW. Điểm cuối bằng 0,5 lần cosine StyleID cộng 0,5 lần cosine của ID512 sau adapter Hard-Positive. Giao diện hiển thị 1.280 chiều và điểm từng nhánh.

Ngưỡng mặc định 0,30 tương ứng cột Acc@0.3 của bảng LFW. Có thể đổi ngưỡng và chuyển sang Dual-Branch, Disentangle ID512 hoặc Disentangle + LEACE sau khi tính, các điểm đã tính được dùng lại. API `/api/compare` mặc định chọn `mode=dual_hp`; giá trị `mode` còn hỗ trợ `dual`, `raw`, `leace`. Trường `use_leace` của API trước được giữ để các lời gọi cũ tiếp tục chọn raw hoặc LEACE.

Có thể upload ảnh của mình. Preset ảnh cần tài nguyên riêng qua `STYLEID_RESOURCES`. `HF_HOME` đặt vị trí cache tải model. Biến môi trường hệ thống được ưu tiên hơn `.env`.
