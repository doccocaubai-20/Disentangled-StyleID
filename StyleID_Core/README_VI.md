# Bộ file gọn: train → Soup → kiểm thử → server

## Demo Soup + LEACE (đã bổ sung)

Chạy `mo_server.bat`, chọn hai ảnh và bấm tính. Mục **Chế độ Soup** cho phép chuyển **Soup gốc / Soup + LEACE**, luôn hiển thị cả hai điểm cosine. Chuyển chế độ sau khi đã tính không chạy model lại. Mặc định giữ Soup gốc.

File `assets/soup_leace_operator.npz` là bộ lọc đã kiểm thử, kèm metadata và `demo_leace.py`; không phải weightsSoup mới. Server kiểm hash bộ lọc và hai checkpointSoup trước khi cho chạy, không dùng shard thay thế. Khi chia sẻ thư mục, giữ cả assets và demo_leace.py; vẫn cần hai file Soup trong `checkpoints/soup` như hướng dẫn bên dưới. ZIPcode nay có thêm bộ lọc nhỏ, không có checkpointCLIP/Soup hoặc dataset.

Ngưỡng0.40 chỉ để demo, chưa hiệu chuẩn riêng cho các model/chế độ. Cosine tăng không tự chứng minh dự đoán tốt hơn. LEACE giúp một phần tác vụ stylized nhưng có đánh đổi trênLFW; chế độ chọn không tự nhận biết loại ảnh.

Đây là bản tách riêng mã trong dự án. File gốc vẫn giữ nguyên. Không có lượt train, tạo Soup, benchmark hoặc server nào được chạy khi đóng gói. Không chép checkpoint GB hoặc dataset vào thư mục này.

## Các file chính

| File | Công việc |
|---|---|
| `train_a100_50.py` | Nhánh A100 50%: train gộp shard **[1,2,4,5,6]**, hai epoch; tách phần train từ `a100_script_ref.py`, bỏ phần tự benchmark sau train |
| `train_shard_0_den_6.py` | Nhánh train nối tiếp: chọn **một shard 0…6** bằng `--shard`; giữ checkpoint để chạy shard kế tiếp; có snapshot tên shard |
| `tao_soup.py` | Trộn riêng hai cặp **shard6 + A100 50%**, mặc định 50/50, chạy CPU |
| `test_sksf_a.py` | Kiểm thử StyleID và Soup trên SKSF-A **1.876 cặp** |
| `test_lfw_full_6000.py` | Kiểm thử StyleID và Soup trên LFW **View2 đủ6.000 cặp/10fold**; không dùng lượt dev-test1.000 |
| `server.py` | Web FastAPI so hai ảnh, giữ giao diện từ `web_demo.py` |
| `mo_server.bat` | Windows: bấm đúp để mở server, vào `http://localhost:8000` |
| `requirements.txt` | Thư viện Python; không tự cài khi đóng gói |
| `StyleID_Colab_Check_3Models_LFW6000.py` | Một file upload cho notebook Colab mới: shard6, A10050%, Soup và StyleID; SKSF-A + LFWfull6000; lấy input từ Drive, không train |
| `StyleID_Colab_Check_3Models_LFW6000_v2.py` | Bản cập nhật theo mã fullLFW người dùng cung cấp: batch128, 10fold grid200; ưu tiên dùng bản này cho lượt so sánh mới |
| `archive/` | Hai bản lịch sử khác recipe để không nhầm chúng thành một lượt train |
| `MANIFEST.json` | Nguồn, SHA256 và những điều chỉnh khi tách file |

**“Train đến6” ở đây hiểu là nhánh nối shard đến shard6, không phải train lại cả dãy tự động.** Bản `train_shard_0_den_6.py` dùng trainer nối tiếp hiện có, bỏ mặc định shard7. Lịch sử shard0 chỉ dùng một phần dữ liệu và shard1 có recipe riêng; không khẳng định chạy mới0→6 bằng một recipe sẽ tái tạo đúng checkpoint lịch sử. Bản shard1 gốc được giữ trong `archive/train_shard1_original.py` để truy vết.

Có hai mã A10050 khác nhau trong dự án: bản tham chiếu dùng LR1,2e-4/SupCon hệ số1; bản multi-shard dùng LR1,5e-4/SupCon hệ số0,5 và tự nạp tiếp checkpoint nếu có. File chính chọn **bản tham chiếu**; bản multi-shard được giữ nguyên ở `archive/train_a100_multi_shards_original.py`. Chưa có log đầy đủ để khẳng định một trong hai là bản nguồn duy nhất của weight A100 lịch sử. Không dùng hai bản như cùng một recipe.

## Train trên Colab/A100 — khi thực sự muốn chạy mới

Giữ notebook cũ; chạy riêng file cần dùng, không Run all. Các câu lệnh dưới là hướng dẫn sử dụng, chưa được chạy trong lần tách file này.

```text
python train_a100_50.py --output-dir /content/drive/MyDrive/StyleID_A100_Champion/a100_50_new
python train_shard_0_den_6.py --shard 6 --work-dir /content/drive/MyDrive/StyleID_A100_Champion/shard_seq_new
```

Train A10050 mặc định dùng thư mục làm việc riêng `runs/a100_50`; output ở `checkpoints/a100_50`. Nhánh nối shard dùng `runs/shard_seq`; đặt cặp generic của checkpoint trước vào work-dir nếu muốn học tiếp:

```text
disentangled_lora_clip.pt
disentangled_head.pt
```

Sau shard6 có cả cặp generic cho continuation và snapshot `disentangled_lora_clip_shard6.pt` / `disentangled_head_shard6.pt`. File này không tự tìm parent trên toàn Drive. Chạy lại train sẽ cập nhật output tại đường dẫn đã chọn; nên dùng đường dẫn riêng, không thay Soup đang giữ.

## Tạo Soup riêng — không cần train

Trong `--parent-dir` cần đủ bốn file:

```text
disentangled_lora_clip_shard6.pt
disentangled_head_shard6.pt
disentangled_lora_clip_a100_final.pt
disentangled_head_a100_final.pt
```

```text
python tao_soup.py --parent-dir /content/drive/MyDrive/StyleID_A100_Champion
```

Hoặc dùng `--shard6-backbone`, `--shard6-head`, `--a100-backbone`, `--a100-head` để chỉ các file nằm ở thư mục khác nhau. Output mặc định ở `checkpoints/soup`, không ghi vào cặp Soup gốc tại thư mục dự án. File đã tồn tại sẽ không bị ghi đè trừ khi thêm `--overwrite` rõ ràng. Phép trộn giữ quy tắc50/50 float32 rồi cast dtype như đoạn cũ trong `test_all.py`, kể cả buffer nguyên. Khớp công thức không bảo đảm byte/hash đầu ra giống checkpoint Soup lịch sử nếu parent không đúng bản.

## Kiểm thử và server trên máy hiện tại

Vì `StyleID_Core` nằm trong dự án, các file tự tìm ảnh và Soup ở thư mục cha. Không cần chép weight hoặc `SKSF-A_data` lần hai. Nếu có Soup mới ở `checkpoints/soup`, đường dẫn đó được ưu tiên; muốn dùng đúng Soup gốc thì đặt `STYLEID_CHECKPOINT_DIR` bằng thư mục dự án.

```powershell
cd C:\Users\admin\Documents\StyleID-main\StyleID_Core
$env:STYLEID_CHECKPOINT_DIR = 'C:\Users\admin\Documents\StyleID-main'
python test_sksf_a.py
python test_lfw_full_6000.py
```

Để dùng cache LFW có sẵn, đặt `STYLEID_LFW_HOME` tới thư mục chứa `pairs.txt` và `lfw_funneled/`. Nếu không đặt, script giữ helper tải/cache LFW của scikit-learn từ bản gốc. LFW trong máy hiện tại chưa xác nhận đủ ảnh; đóng gói không tải lại hay chạy lại lượt bạn đã làm.

```powershell
$env:STYLEID_LFW_HOME = 'D:\datasets\lfw_home'
python test_lfw_full_6000.py
```

Mở server bằng `mo_server.bat` hoặc `python server.py`. Server dùng ảnh preset tại `data/` và `SKSF-A_data/` của thư mục tài nguyên. Muốn chuyển folder sang máy khác, mang weight/dataset riêng rồi đặt:

```powershell
$env:STYLEID_RESOURCES = 'D:\StyleID_resources'
$env:STYLEID_CHECKPOINT_DIR = 'D:\StyleID_weights'
python server.py
```

Hai file weight Soup bắt buộc đúng tên `disentangled_lora_clip_SOUP_S6_A100.pt` và `disentangled_head_SOUP_S6_A100.pt`. Bản tách không tự thay bằng shard6 khi thiếu Soup. Benchmark vẫn giữ cách tiền xử lý/tính metric của mã nguồn cũ; đây không phải một lần tái xác nhận kết quả công bố. Oracle Best Acc và ROC trên test khác accuracy10fold; không dùng lẫn.

Các script lịch sử có thể tải model/dữ liệu khi được chạy; chỉ đọc hoặc `--help` của các file train/Soup không khởi động các lượt đó. Chưa kiểm end-to-end trên GPU/server trong lần đóng gói; đã kiểm cú pháp, CLI và các quy tắc đường dẫn/giao thức mà không nạp model thật.
