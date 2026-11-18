# Báo cáo bàn giao Guide v2 — 13/11/2026

## Cập nhật 17/11/2026

- Đã thêm chấm nhiều video theo manifest validation/test. Báo cáo cộng sự kiện và thời lượng thông thoáng, kiểm tra video trùng, kiểm tra SHA-256 và từ chối gộp các lần chạy dùng cấu hình so sánh khác nhau. Tệp nhãn đời trước có thể đối chiếu theo tên video.
- Phát lại kiểm tra hạn của từng khung ngay khi tới lượt model. Kết quả ngữ cảnh cũ bị loại nếu quan sát mới hơn mất tin cậy hoặc thấy vật cản chặn hướng. Thời lượng nguồn dùng để chấm được ghi riêng với thời gian xử lý cuối phiên.
- Thử lại 6 giây trên CPU. Đọc 180 khung, thay 158 khung đang chờ, bỏ 2 khung trước khi nộp tác vụ và bỏ 10 lượt gọi model đã hết hạn sau khi đợi thiết bị. 5 tác vụ ngữ cảnh bị loại. Bản chạy xuất 31 quyết định UNCLEAR và không ghi lỗi nạp model. Xem `runs/guide_v2/replay_nov17/report.json`. Đây là đoạn thử chưa gán nhãn.
- Gói `dist/guide_kaggle_v2_models.zip` chứa mã, trọng số và giọng đọc. Gói đã được tải vào Dataset Kaggle riêng tư [Guide v2 Models](https://www.kaggle.com/datasets/nguyenduchng/guide-v2-models).

## Kết quả chạy Kaggle GPU — 17/11/2026

Bản [Guide v2 GPU Demo, phiên 1](https://www.kaggle.com/code/nguyenduchng/guide-v2-gpu-demo?scriptVersionId=370386489) đã chạy xong và được lưu ở chế độ riêng tư. Cấu hình phiên là GPU T4 x2. Notebook giới hạn xử lý ở thiết bị `cuda:0`, nhận dạng là Tesla T4. PyTorch `2.11.0+cu130` xác nhận CUDA hoạt động. YOLOv8n ROD, SegFormer B0 và Depth Anything V2 Small đều qua bước thử một khung hình.

| Chỉ số | Kết quả |
|---|---:|
| Khung hình đã xử lý trên clip demo 6 giây | 60 |
| Tốc độ xử lý và xuất kết quả | 4,42 khung/giây |
| p50 xử lý nhận diện, theo dõi và ra quyết định | 139,6 ms |
| p95 xử lý nhận diện, theo dõi và ra quyết định | 166,8 ms |
| Bộ nhớ GPU đỉnh do PyTorch cấp phát | 415,6 MiB |

Mở [Output của phiên 1](https://www.kaggle.com/code/nguyenduchng/guide-v2-gpu-demo/output?scriptVersionId=370386489), vào `guide_demo_20261116_182430` để xem `demo.mp4`, `report.json`, nhật ký và âm thanh. Video MP4 hiển thị được trong trình duyệt với thời lượng khoảng 6 giây. Cả 60 quyết định của clip này là `SLOW`. Clip chưa có nhãn. Các chỉ số trên đo chế độ offline, nên mục độ trễ từ camera đến thiết bị âm thanh vẫn chưa có số đo.

## Phần đã thực hiện

- Lõi độc lập `guide_core`, dùng chung cho các đầu vào tương lai; notebook mới gọi lõi, không sao chép thuật toán.
- Nạp YOLOv8n ROD, YOLO26n, SegFormer B0 ADE20K và Depth Anything V2 Small thành công trên môi trường CPU hiện có.
- Kết hợp cùng khung hình, giữ nguồn model; BoT-SORT/GMC; nền/chắn/chưa rõ; tìm hành lang liên tục; xác nhận hướng theo thời gian nguồn; hủy hướng bị chắn.
- Âm thanh Khánh Vy có bộ clip đúng nghĩa lệch nhẹ/giữ hướng; một câu mỗi thời điểm, câu chờ mới nhất, hủy điều lệnh mất hiệu lực.
- Xuất MP4 H.264/AAC, phụ đề, semantic/depth, nhật ký quyết định/âm thanh, báo cáo hiệu năng.
- Công cụ gán và chấm sự kiện, kiểm tra chia tập theo phiên/địa điểm, so sánh cùng khung hình và gói Kaggle.
- Desktop/web cũ giữ nguyên đường chạy làm bản đối chiếu.

## Phép đo CPU đã chạy

Nguồn: `data/guide_demo/val1_excerpt.mp4`, 6 giây trích từ video dự án. Mỗi cấu hình dùng **cùng 60 khung**, detector 480, semantic 512, depth 518. Không có nhãn sự kiện. Đây là smoke test ngắn; không phải kết quả trên Kaggle GPU hoặc thử nghiệm đi bộ.

| Cấu hình | Tốc độ xử lý/xuất (khung/s) | p95 nhận diện + tracking + quyết định (ms) |
|---|---:|---:|
| YOLOv8n riêng | 4,43 | 132,4 |
| YOLO26n riêng | 4,43 | 130,5 |
| YOLOv8n + SegFormer | 0,94 | 1042,6 |
| YOLOv8n + SegFormer + depth | 0,57 | 1868,7 |
| Hai detector + SegFormer + depth | 0,54 | 1899,6 |

Tốc độ ở bảng gồm vẽ/ghi bản xem trước và ghép WAV, không gồm nạp model và lần mã hóa H.264 cuối. Nó **không phải FPS webcam**, cũng không phải FPS suy luận YOLO đơn. Ví dụ ở cấu hình ba model, median thời gian từng model khoảng 82,9 ms YOLO, 794,7 ms SegFormer, 685,8 ms depth.

Dữ liệu thô: `runs/guide_v2/comparison_cpu/comparison.json` và `report.json` trong từng thư mục. Video ba model: `runs/guide_v2/comparison_cpu/plus_depth/demo.mp4`.

Đoạn này phần lớn cho SLOW (59 khung) và một UNCLEAR ở các cấu hình có semantic. Detector riêng trả UNCLEAR khi chưa có bằng chứng mặt nền. **Không dùng sự thận trọng đó như bằng chứng chỉ dẫn tốt**: vẫn cần đo tỷ lệ chỉ được hướng đúng trên các khoảng đã gán nhãn.

Chưa chứng minh lợi ích chất lượng khi bật detector thứ hai; giữ tắt mặc định. Chưa thay model desktop.

## Phát lại theo thời gian thực

Đã chạy video 6 giây với cả ba model trên CPU:

- 180 khung được giải mã; 173 khung bị thay bằng khung mới hơn trong ô đệm.
- 1 khung quá hạn trước suy luận; 2 kết quả context quá hạn bị bỏ.
- Không có lỗi nạp model.
- Đầu ra là UNCLEAR, không phát hướng từ context đã quá 700 ms.
- RAM lấy mẫu đạt khoảng 938 MiB; môi trường này không có CUDA khả dụng.
- `frame_to_decision_ms` trong report hiện đo tuổi khung gắn với mọi đầu ra, **bao gồm watchdog báo mất dữ liệu**; không dùng p95 của mục này như độ trễ gửi cảnh báo khẩn.
- Chưa đo được thời điểm gửi loa thật; trường tương ứng là null. Âm thanh đã được xếp và ghép vào MP4, không phát trực tiếp từ notebook.

Đầu ra: `runs/guide_v2/replay_cpu/`. Phép thử này xác nhận cơ chế từ chối kết quả cũ; không đạt mục tiêu tốc độ sử dụng trực tiếp trên CPU.

## Kiểm tra

42 kiểm thử riêng cho lõi mới đã đạt, gồm: sai frame/session/geometry, sửa ảnh gương, hợp nhất nguồn, semantic/depth thiếu hoặc hỏng, lối đi không liên tục, cửa, đường/floor theo bối cảnh, lối quá hẹp, trái/phải đối xứng, vật cản cắt ngang, khoảng xác nhận 0,4 giây, dữ liệu cũ, model lỗi/chậm, track thực, camera mất quan sát, câu chờ bị thay/hết hạn, điều lệnh đang phát bị hủy, không chồng âm thanh và chấm hướng đang đọc đi vào vùng nguy hiểm.

Kiểm thử web cũ ban đầu bị Windows chặn socket localhost; chạy lại với quyền truy cập localhost thì 14 kiểm thử runtime đều đạt. Nhật ký đầy đủ ở `runs/guide_v2/final_test_results.txt`.

Lần kiểm thử toàn dự án cuối: 78 ca, 77 đạt và 1 được bỏ qua theo điều kiện có sẵn. Các chỉ số âm thanh trong phép đo CPU ban đầu chưa có hàng chờ đáng kể; lõi đã bổ sung kiểm thử riêng để giữ đúng thời gian chờ ban đầu khi bằng chứng của câu chờ được làm mới.

Các ô Python của notebook đã kiểm tra cú pháp. ZIP đã qua kiểm tra toàn vẹn trên CPU và trong notebook Kaggle. Phiên GPU đã cài thư viện, qua kiểm tra tương thích và lưu đầy đủ kết quả của clip ngắn.

## Chưa nghiệm thu

Chưa có bộ nhãn độc lập đủ 100 sự kiện nguy hiểm và 30 phút thông thoáng; chưa có phép thử đầy đủ hẻm/lối hẹp/xe đi ngang/bậc/tối/rung và camera liên tục 30 phút. Vì vậy chưa công bố recall ≥90%, cảnh báo sai ≤0,2/phút, không có hướng sai trên test, hay p95 gửi loa ≤700 ms.

Bước tiếp theo để nghiệm thu là gán nhãn các video validation/test độc lập, chạy các cấu hình trên cùng dữ liệu và đo riêng độ trễ cảnh báo trong chế độ thời gian thực.

