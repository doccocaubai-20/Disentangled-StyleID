# Huấn luyện và trọng số trên Drive

## Bốn thư mục checkpoint

[Thư mục trọng số trên Google Drive](https://drive.google.com/drive/folders/11DPH1P4QDKjiZwaAzJsdigK2AAPQezD1)

| Thư mục | File |
|---|---|
| [no_ortho](https://drive.google.com/drive/folders/1MCeXXLq8qWo89KZoU-bORtNYxWP5Deod) | `disentangled_lora_clip_no_ortho.pt` và `disentangled_head_no_ortho.pt` |
| [ortho](https://drive.google.com/drive/folders/1cjxJQxkEOzlXWHWw6K1G-Z_3k3VE50aE) | `disentangled_lora_clip_a100_final.pt` và `disentangled_head_a100_final.pt` |
| [soup](https://drive.google.com/drive/folders/1SrGqDecmDIpsqqIP_sq_glpbh6t8hMRZ) | `disentangled_lora_clip_SOUP_S6_A100.pt` và `disentangled_head_SOUP_S6_A100.pt` |
| [hard-positive-adapter](https://drive.google.com/drive/folders/1Id5FCJer80UQM8o_FZvXoLJ71opmzyqX) | [lfw_hard_positive_adapter.pt](https://drive.google.com/file/d/12saHW4q3RPIVwel2uJpsnsgv_YJDZLA8/view) |

`no_ortho` chứa bản sao đúng lượt chạy benchmark `id_style_no_ortho_20261024T131324809914Z` với 111.998 ảnh, 4.073 danh tính và 10 phong cách. `ortho` chứa bản sao của checkpoint A100 huấn luyện có trực giao trên cùng nhóm shard. Dòng Disentangle ID512 trong bảng cuối dùng checkpoint trong `soup`.

Benchmark và demo tải StyleID gốc từ `kwanY/styleid`. Dual-Branch được tạo khi suy luận từ StyleID 768D và Soup ID512, mỗi nhánh đóng góp 50% cosine. Trên LFW, adapter rank 32 được áp dụng cho ID512 trước khi ghép. File adapter chứa trọng số head và metadata khóa phiên bản, SHA256 của Soup và các revision mô hình.

## Huấn luyện đối chứng không trực giao

```text
python 01_train/train_no_ortho.py --output-dir checkpoints/no_ortho --work-dir runs/no_ortho
```

Mặc định script dùng seed 20261024, batch 128, hai epoch và gộp shard [1,2,4,5,6]. Nó khởi tạo CLIP, LoRA và head mới, rồi tối ưu loss ID cộng 0,5 lần loss style. BF16 được chọn tự động trên GPU CUDA hỗ trợ, các thiết bị còn lại dùng FP32. Có thể chỉ định `--device`, `--precision`, `--batch-size` hoặc dùng `--data-dir` trỏ tới dữ liệu đã giải nén có cấu trúc `model/style/*.png`.

Mỗi lần chạy tạo thư mục mới trong `--output-dir`, ghi backbone, head, `recipe.json` và `completed.json`. Các file tải và giải nén trong thư mục tạm của lượt chạy được dọn sau khi lưu xong. Có thể trỏ `--output-dir` đến thư mục Drive đã mount.

## Huấn luyện có trực giao và tạo Soup

`01_train/train_a100_50.py` gộp shard [1,2,4,5,6] trong hai epoch và cần GPU CUDA hỗ trợ BF16. `01_train/train_shard_0_den_6.py` chạy tiếp một shard mỗi lần. Các lượt đầu của lịch sử S6 có thiết lập riêng. Dùng checkpoint S6 đã lưu nếu cần tái lập đúng Soup lịch sử.

```text
python 01_train/train_a100_50.py --output-dir checkpoints/a100_50
python 01_train/train_shard_0_den_6.py --shard 6 --work-dir runs/shard_seq
python 02_soup/model_soup_averaging.py --parent-dir checkpoints/parents
```

Lệnh shard 6 cần cặp checkpoint tiếp nối của shard 5 trong work-dir. Thư mục cha cho Soup cần bốn tệp sau.

```text
disentangled_lora_clip_shard6.pt
disentangled_head_shard6.pt
disentangled_lora_clip_a100_final.pt
disentangled_head_a100_final.pt
```

Soup mặc định trộn 50/50 và xuất vào `checkpoints/soup/`. Notebook benchmark và web demo kiểm tra SHA256 để nhận diện đúng cặp Soup đã công bố.
