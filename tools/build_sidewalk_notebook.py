"""Build the self-contained Kaggle notebook from the maintained training script."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"sidewalk-{len(cells):02d}", "metadata": {},
                "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
# SegFormer Sidewalk — Kaggle, giảm bộ nhớ

Notebook độc lập để sửa phiên SegFormer bị hết bộ nhớ. Giữ SegFormer-B0 và 35 ID nhãn
của `segments/sidewalk-semantic`; ID 0 và giá trị nguồn 255 được bỏ qua trong loss/metric.
Nguồn: https://huggingface.co/datasets/segments/sidewalk-semantic

## Chuẩn bị

1. Tạo notebook Kaggle mới và import file `.ipynb` này. Dùng **kernel/session mới**.
2. Bật Internet và GPU. Nếu Kaggle cấp T4 x2, code chỉ dùng GPU số 0.
3. Trong **Add-ons → Secrets**, tạo/bật `HF_TOKEN`. Token phải thuộc tài khoản đã được
   cấp quyền đọc dataset. Không dán token vào code hoặc output.
4. Chạy lần lượt các ô từ trên xuống. Ô đầu cài phiên bản Transformers cố định;
   không cài lại PyTorch của Kaggle.

Mặc định `RUN_MODE="smoke"`: 20 optimizer steps trên 64 ảnh train, 16 validation,
16 test. Đây là phép thử train → validation → lưu checkpoint → tải lại trọng số tốt
nhất → xuất mô hình, chưa phải kết quả huấn luyện đầy đủ.
""")
    add("code", """
import os
import sys
import subprocess

if any(name in sys.modules for name in ("torch", "transformers", "datasets")):
    raise RuntimeError("Hãy restart kernel/session rồi chạy lại từ ô đầu để giải phóng bản mô hình cũ.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.57.1", "accelerate>=1.2,<2", "datasets>=3,<5", "psutil",
])
print("Đã chuẩn bị thư viện. Tiếp tục chạy các ô bên dưới.")
""")
    add("markdown", """
## Chọn chế độ chạy

- Lần đầu giữ `RUN_MODE="smoke"`. Khi thấy `SMOKE_COMPLETED`, cả train, validation,
  checkpoint và export đã đi hết luồng.
- Sau đó đổi thành `RUN_MODE="full"`, restart kernel/session rồi chạy lại từ đầu.
  Chế độ full dùng 800/100/100 ảnh và mặc định 10 epoch.
- Ảnh 384 × 384, batch train/eval 1, gradient accumulation 4. Sau khi ổn định mới
  cân nhắc thử 512. GPU thứ hai không được dùng; số GPU mà Kaggle cấp vẫn có thể là 2.
- Giữ `RESUME_CHECKPOINT=None` cho lần chạy mới. Đặt `True` để tiếp tục checkpoint
  mới nhất trong cùng thư mục, hoặc nhập đường dẫn checkpoint đã thêm từ Kaggle Input.
  Khi nhập Input cần có cả `run_config.json` ở thư mục cha và các file trạng thái đầy đủ.

Train/validation/test giữ đúng cách chia ngẫu nhiên hai bước của notebook bạn gửi.
Đây là baseline nội bộ dataset; để đánh giá đi bộ thực tế cần video/tuyến đường độc lập.
Các chỉ số của bản 384 không so trực tiếp với bản 512 như thể cùng cấu hình.
""")
    add("code", """
RUN_MODE = "smoke"           # Sau khi chạy thử thành công, đổi thành "full".
IMAGE_SIZE = 384
EPOCHS = 10
OUTPUT_ROOT = "/kaggle/working"
RESUME_CHECKPOINT = None     # True hoặc đường dẫn đầy đủ để tiếp tục bản notebook mới.
""")
    add("markdown", """
## Định nghĩa pipeline

Ô dưới chỉ khai báo hàm và lớp. Ảnh/mask được thu nhỏ từng ảnh rồi lưu trên ổ đĩa;
DataLoader đọc batch hiện tại. Validation chuyển logits thành ma trận đếm 35 × 35
ngay từng batch và cộng số đếm để tính IoU toàn tập, không lưu toàn bộ logits/mask.
Tham khảo API: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer
""")
    script = (ROOT / "tools" / "train_sidewalk_segformer_kaggle.py").read_text(encoding="utf-8")
    add("code", script.split('\nif __name__ == "__main__":', 1)[0])
    add("markdown", """
## Chạy train, validation và lưu mô hình

Chế độ smoke lưu/evaluate mỗi 10 optimizer steps; full mỗi 100 steps. Dòng `MEMORY`
ghi RAM tiến trình, giới hạn RAM container nếu đọc được, VRAM đang dùng/đỉnh sử dụng.
`memory_log.jsonl` và checkpoint đã lưu giúp kiểm tra khi phiên bị dừng.

Khi đổi từ smoke sang full, restart rồi Run All. Có thể dùng Save Version → Save & Run All
để Kaggle chạy bản full sau khi đã xác nhận smoke thành công.
""")
    add("code", """
summary = run_training(
    mode=RUN_MODE,
    output_root=OUTPUT_ROOT,
    image_size=IMAGE_SIZE,
    epochs=EPOCHS,
    resume_checkpoint=RESUME_CHECKPOINT,
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
""")
    add("markdown", """
## Nhận biết đã xong và tải kết quả

- **Smoke:** kết thúc bằng `SMOKE_COMPLETED`; chỉ xác nhận luồng chạy.
- **Full:** kết thúc bằng `TRAINING_COMPLETED`, sau đó Kaggle Version báo thành công.
- Mô hình full: `/kaggle/working/segformer_sidewalk_best/` và
  `/kaggle/working/segformer_sidewalk_best.zip`.
- Bộ mô hình gồm `model.safetensors`, `config.json`, `preprocessor_config.json`,
  `id2label.json`, `run_config.json`, `training_summary.json`.
- Checkpoint đầy đủ: `/kaggle/working/segformer_sidewalk_full_checkpoints/checkpoint-*/`.
  Tải cả output nếu muốn resume, không chỉ tải ZIP mô hình.
- Để báo lỗi, gửi 10 dòng `MEMORY` cuối cùng và traceback cuối. Nếu kernel bị kill,
  file `FAILED.json` có thể không được ghi; các dòng log/checkpoint trước đó vẫn có ích.

Trọng số SegFormer ở định dạng Hugging Face; dùng trong pipeline semantic segmentation.
Detector ROD `best.pt` của ứng dụng là mô hình riêng. Notebook này chỉ huấn luyện SegFormer.
""")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    destination = ROOT / "kaggle_train_sidewalk_segformer_low_memory.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build_notebook())
