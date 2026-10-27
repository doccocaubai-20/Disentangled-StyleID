# Mã Colab và vị trí trọng số trên Drive

GitHub chứa mã nguồn. Các checkpoint `.pt` lớn được lưu trên Google Drive của người chạy Colab; repo không chứa các checkpoint này.

## Các file mã

| File | Chức năng |
|---|---|
| `01_train/train_shard_0_den_6.py` | Recipe train tuần tự theo shard của bản lịch sử |
| `01_train/train_a100_50.py` | Recipe lịch sử có loss trực giao, gộp năm shard [1,2,4,5,6] |
| `02_soup/tao_soup.py` | Kết hợp trọng số S6 và A100; không phải lượt train |
| `01_train/colab/Dong2_Train_ID_Style_Khong_Ortho.py` | Bản đã dùng train trên Colab: ID512 + style256, bỏ loss trực giao, hai epoch trên năm shard gộp chung |
| `01_train/colab/Dong4_LEACE_Tu_Checkpoint_Dong3.py` | Fit và đánh giá LEACE từ cache vector đã benchmark; chạy CPU |

Hai file Colab được sao chép nguyên byte từ mã đã lưu trong workspace, không thay đổi công thức train hay quy trình lọc. Các pilot teacher, fusion và distillation không thuộc bộ mã này.

## Chạy bản không trực giao

Chọn GPU hỗ trợ BF16, khuyến nghị A100. Cài thư viện trước khi import Transformers:

```python
%pip install -q transformers==4.52.0 huggingface_hub Pillow tqdm scikit-learn
```

Upload đúng file `Dong2_Train_ID_Style_Khong_Ortho.py`, sau đó chạy:

```python
import runpy
runpy.run_path('/content/Dong2_Train_ID_Style_Khong_Ortho.py', run_name='__main__')
```

Mặc định: seed 20261024, batch 128, hai epoch, gộp shard [1,2,4,5,6], khởi tạo OpenAI CLIP cùng LoRA/head mới. Chạy lại tạo một lượt train mới, không resume. File tự ghi checkpoint và metadata lên Drive; dữ liệu ảnh tạm trên Colab được dọn sau khi lưu thành công.

## Trọng số đã lưu ở đâu

Trong giao diện Google Drive, bắt đầu từ **Drive của tôi**. Đường dẫn dưới đây lấy từ mã và metadata lượt chạy đã nhận; chưa phải URL chia sẻ.

### Mô hình chính Disentangle có nguồn gốc Soup

Mở thư mục **StyleID_A100_Champion**. Hai file cần lấy cùng nhau:

```text
disentangled_lora_clip_SOUP_S6_A100.pt
disentangled_head_SOUP_S6_A100.pt
```

Trong cùng thư mục, checkpoint cha lịch sử có tên:

```text
disentangled_lora_clip_shard6.pt
disentangled_head_shard6.pt
disentangled_lora_clip_a100_final.pt
disentangled_head_a100_final.pt
```

### Mô hình không trực giao đã train ngày 24 tháng 10

Mở lần lượt:

```text
StyleID_NoOrtho/
  id_style_no_ortho_20261024T131324809914Z/
    disentangled_lora_clip_no_ortho.pt
    disentangled_head_no_ortho.pt
    recipe.json
    completed.json
```

Nếu có nhiều thư mục bắt đầu bằng `id_style_no_ortho_`, chọn đúng tên trên để lấy checkpoint khớp kết quả đã gửi. Lượt này ghi nhận 111.998 ảnh, 4.073 danh tính và 10 phong cách. `benchmark_5models_fresh_20261024T162806060605Z` ở bên trong là kết quả kiểm thử, không thay thế hai file trọng số train.

### Bộ lọc LEACE

Demo dùng `assets/soup_leace_operator.npz` cùng `assets/soup_leace_metadata.json` đã có trong repo. Bộ lọc nhỏ này không thay thế backbone/head: để chạy Disentangle + LEACE cần cả hai checkpoint Soup và bộ lọc.

File fit LEACE Colab mặc định đọc cache tại:

```text
/content/drive/MyDrive/StyleID_A100_Champion/benchmark_3models_full_lfw_v2
```

Nó xuất `row4_leace_operator.npz` và kết quả vào thư mục `row4_leace_<thời gian>` cạnh thư mục cache. Chỉ tìm thư mục này nếu đã thực sự chạy file fit đó. Bộ lọc demo được cung cấp sẵn trong repo nên không cần chạy lại để lấy bộ lọc.

## Lấy link để gửi nhóm

1. Trong Google Drive, nhấp phải thư mục hoặc file cần gửi, chọn **Chia sẻ**.
2. Cấp quyền xem/tải cho thành viên nhóm theo nhu cầu.
3. Chọn **Sao chép đường liên kết**, rồi gửi kèm link mã nguồn GitHub.

Nên gửi link của thư mục chứa đủ cặp backbone và head. Đối với bản không trực giao, gửi kèm `recipe.json` và `completed.json` để đối chiếu lượt train. Ảnh chụp tên thư mục không chứa ID thư mục nên không thể suy ra URL chia sẻ chính xác từ ảnh đó.
