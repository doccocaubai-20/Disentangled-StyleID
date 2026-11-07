# Huấn luyện và trọng số trên Drive

## Checkpoint dùng cho kết quả cuối cùng

| Cấu hình | File backbone và head |
|---|---|
| ID + Style không trực giao | `disentangled_lora_clip_no_ortho.pt` và `disentangled_head_no_ortho.pt` |
| Disentangle ID512 | `disentangled_lora_clip_SOUP_S6_A100.pt` và `disentangled_head_SOUP_S6_A100.pt` |
| StyleID | Model chính thức `kwanY/styleid`, revision khóa trong notebook |
| Dual-Branch | Kết hợp biểu diễn StyleID 768D và Disentangle 512D, trọng số 50/50 |
| Dual-Branch + Hard-Positive | Thêm adapter vào nhánh ID512, trọng số adapter đã nhúng trong notebook kiểm chứng |

[Mở thư mục Soup trên Drive](https://drive.google.com/drive/folders/11DPH1P4QDKjiZwaAzJsdigK2AAPQezD1). Giữ cả backbone và head cùng nhau.

[Mở thư mục bản không trực giao trên Drive](https://drive.google.com/drive/folders/1W-mlpZHZIANjtu5bPiDTZsHCk9VmRKnE). Lượt train dùng cho bảng có thư mục `id_style_no_ortho_20261024T131324809914Z` trong `StyleID_NoOrtho`, ghi nhận 111.998 ảnh, 4.073 danh tính và 10 phong cách.

Các liên kết trên cũng là nguồn tải được cấu hình trong notebook đã chạy. Chủ sở hữu Drive quản lý quyền xem, tải và hạn mức.

## Huấn luyện đối chứng không trực giao

Mở `01_train/colab/Dong2_Train_ID_Style_Khong_Ortho.ipynb` trên Colab và chọn A100 hỗ trợ BF16. Notebook tự tạo file Python trước khi chạy train.

Recipe dùng CLIP mới, LoRA và head mới, ID512 và style256, loss ID cộng 0,5 lần loss style. Loss trực giao được tắt. Seed 20261024, batch 128, hai epoch, gộp năm shard [1,2,4,5,6]. Checkpoint và metadata được lưu vào thư mục lượt chạy mới trên Drive, dữ liệu tạm được dọn sau khi lưu thành công.

Nếu chạy file Python riêng, cài thư viện trước.

```python
%pip install -q transformers==4.52.0 huggingface_hub Pillow tqdm scikit-learn
import runpy
runpy.run_path('/content/Dong2_Train_ID_Style_Khong_Ortho.py', run_name='__main__')
```

## Hai nhánh có trực giao và tạo Soup

`01_train/train_a100_50.py` là recipe gộp shard [1,2,4,5,6], hai epoch. `01_train/train_shard_0_den_6.py` chạy tiếp một shard mỗi lần. Lịch sử checkpoint S6 có các lượt đầu khác recipe trainer chung, vì vậy cần checkpoint đã lưu để kiểm chứng đúng kết quả cuối.

```text
python 01_train/train_a100_50.py --output-dir /content/drive/MyDrive/StyleID_A100_Champion/a100_50_new
python 01_train/train_shard_0_den_6.py --shard 6 --work-dir /content/drive/MyDrive/StyleID_A100_Champion/shard_seq_new
python 02_soup/tao_soup.py --parent-dir /content/drive/MyDrive/StyleID_A100_Champion
```

Lệnh shard 6 cần cặp checkpoint generic của shard trước trong work-dir. Tạo Soup yêu cầu bốn file cha.

```text
disentangled_lora_clip_shard6.pt
disentangled_head_shard6.pt
disentangled_lora_clip_a100_final.pt
disentangled_head_a100_final.pt
```

Soup mặc định trộn 50/50 và xuất vào `checkpoints/soup/`. Parent khác phiên bản có thể tạo checkpoint khác kết quả đã nhận. Notebook kiểm chứng khóa SHA256 để nhận diện đúng các checkpoint.

## Adapter và bộ lọc dùng cho demo

Demo mặc định dùng **Dual-Branch + Hard-Positive**, ngưỡng 0,30. `assets/lfw_hard_positive_adapter.npz` chứa đúng hai ma trận adapter đã nhúng trong notebook kiểm chứng Colab. `assets/lfw_hard_positive_metadata.json` khóa SHA256, revision của StyleID và CLIP, cùng cặp checkpoint Soup.

Điểm cuối bằng 0,5 cosine StyleID cộng 0,5 cosine ID512 sau adapter. Hai nhánh đã chuẩn hóa tương đương vector ghép 1.280 chiều. Chạy demo cần hai checkpoint Soup ở trên và truy cập model StyleID.

Chế độ LEACE vẫn đọc `assets/soup_leace_operator.npz` cùng `assets/soup_leace_metadata.json`. Đây là bộ lọc LEACE lịch sử của Soup, được chọn riêng qua menu cấu hình.

## Chia sẻ cho nhóm

Gửi liên kết repository cùng thư mục Drive chứa đủ cặp backbone và head. Với lượt train mới, gửi thêm `recipe.json` và `completed.json` để xác định đúng dữ liệu, cấu hình và checkpoint.
