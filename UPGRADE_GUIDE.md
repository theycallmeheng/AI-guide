# Nâng cấp detection và cảnh báo

## Trạng thái và cấu hình

### Chế độ CPU nhanh

Chọn `models/results/best.onnx` và ảnh AI 480 trong ứng dụng. ONNX CPU dùng
8 luồng suy luận, PyTorch/OpenCV dùng 1 luồng để giảm tranh chấp CPU. Có thể đặt
`GUIDE_ORT_THREADS=4` để so sánh trên máy khác. Chỉ áp dụng khi backend thực sự
dùng CPUExecutionProvider; không thay cấu hình GPU. Model mặc định vẫn giữ nguyên.
Chạy `python tools/tune_onnx_cpu.py --source data/videos/val1.mp4 --samples 300`
để đo lại. FPS benchmark không bao gồm camera mạng, UI, tracking và tiếng nói.

Runtime dùng detection + track IoU quan sát thực, tối đa một cảnh báo chờ.
Không tự đổi model mặc định dựa trên benchmark tốc độ. Các giá trị trong
`runtime_policy.json` là ngưỡng ban đầu, chưa được hiệu chỉnh trên video iPhone có nhãn.
Gần/vừa/xa là heuristic hình ảnh, không phải khoảng cách mét.
`Manhole` được đọc là miệng cống, không kết luận cống đang mở.
Mặc định tắt rẽ/đi thẳng; các API phân đoạn cũ chỉ còn phục vụ tương thích và kiểm thử.

Âm thanh hiện tại đọc hết, chỉ câu chờ còn được quan sát trong 700 ms mới được phát.
Stop/tắt âm thanh vẫn hủy ngay. `queue_wait_ms` đo từ âm thanh sẵn sàng đến gửi thiết bị;
`dispatch_ms` gồm cả tổng hợp/chờ. Các số này không đo độ trễ loa vật lý.
Session report có `audio_events`; đồng hồ tính từ lúc bắt đầu phiên, cần căn chỉnh
với video trước khi dùng để chấm sự kiện. Không lấy yêu cầu xếp hàng làm cảnh báo đã phát.

## Dữ liệu cần thu

Quay ít nhất 10 phiên ở 5 địa điểm bằng đúng góc iPhone/DroidCam dự kiến.
Chốt ba địa điểm train, một val, một test trước khi trích ảnh. Không chia ngẫu nhiên
khung hình. Chọn 1.000–2.000 ảnh có cảnh trống, vật thấp, người/xe, cột, bậc,
lề đường, rung, tối; gán bounding box đầy đủ bằng công cụ hỗ trợ YOLO detection.
Không dùng đa giác segmentation làm bounding box nếu chưa chuyển đổi có kiểm tra.

Giữ thứ tự 25 lớp của `models/results/data.yaml`. Ghép ảnh ROD và iPhone dưới
`images/{train,val,test}/{source}/...`, nhãn dưới `labels/{train,val,test}/{source}/...`;
dùng tên nguồn để tránh ghi đè. Bảo toàn split ROD; không tuyên bố split ROD độc lập
theo tuyến khi không có metadata nguồn. Test iPhone độc lập được báo cáo riêng với ROD.
Manifest CSV có các cột:

```csv
image,split,location_id,session_id,verified_empty
images/train/iphone/location_a_001.jpg,train,location_a,session_a1,0
```

Không bịa location/session. Với ROD thiếu provenance, thêm cột `source=rod`,
`provenance=official_split` để giữ split gốc và ghi rõ số ảnh không xác minh được
độc lập theo tuyến trong kết quả audit. Với ảnh iPhone, location/session bắt buộc.
Ảnh không có vật thể vẫn cần nhãn TXT rỗng và `verified_empty=1` sau khi người gán nhãn kiểm tra.

```powershell
python tools/audit_detection_dataset.py --data data/combined/data.yaml --manifest data/combined/manifest.csv
```

Audit kiểm tra ảnh lỗi, nhãn 5 số, class ID, tọa độ, trùng pixel và location/session lọt nhiều split.
Ảnh gần giống sau crop/nén phải rà soát thêm; audit không chứng minh provenance khai báo là đúng.

## Huấn luyện Kaggle

Import `kaggle_finetune_detection.ipynb`, bật GPU và Internet, sửa bốn đường dẫn
DATA, MANIFEST, MODEL_V8, MODEL_26 ở ô cấu hình theo Input thực tế.
Chạy audit trước. Notebook chứa code độc lập, không cần clone repo.
Chạy smoke một epoch trước, sau đó đổi EPOCHS=50 và chạy phiên mới.
Hai model train lần lượt, batch 4, ảnh 640, seed 42, chỉ validation để chọn checkpoint.
Không chấm test trong vòng lựa chọn checkpoint/ngưỡng. Lưu bản best và kết quả validation
trong `/kaggle/working/detection_candidates`. Giữ phiên bản thư viện và data manifest trong báo cáo.

## Chấm cảnh báo

Tạo JSON truth với thời gian giây theo video và JSON alerts là danh sách tiếng thực sự đã phát:

```json
{"duration_s":60,"events":[{"id":"e1","start_s":2,"end_s":5,"action":"stop","direction":"phía trước","class_name":"Person"}],"clear_intervals":[[10,60]]}
```

```json
[{"time_s":2.3,"action":"stop","direction":"phía trước","class_name":"Person"}]
```

```powershell
python tools/evaluate_alert_events.py --truth truth.json --alerts emitted_alerts.json
python tools/benchmark_runtime.py --source data/videos/val1.mp4 --model models/results/best.pt --model models/results/yolo26n_best.pt --samples 300 --no-render
```

Chuẩn hóa tên lớp/hướng giống nhau trong truth và alerts. Bộ chấm chỉ chấm các khoảng
được gán nhãn; một cảnh báo không được tính trúng nhiều sự kiện. Cảnh báo nhắc lại có
thể là chủ ý cooldown nên số repeated_alerts cần xem cùng video.
Ít hơn 100 sự kiện hoặc 30 phút cảnh trống sẽ mang trạng thái preliminary.

Chọn ứng viên trên validation đạt recall sự kiện >=90%, cảnh báo sai <=0,2/phút,
p95 gửi âm thanh khẩn khi loa rảnh <=700 ms. Trong nhóm đạt, ưu tiên p95 thấp nhất;
nếu chênh dưới 10%, chọn recall cao hơn. Chốt cấu hình rồi đánh giá test độc lập
và chạy camera 30 phút; chỉ thay model mặc định sau khi các kiểm tra này đạt.
Benchmark inference không thay thế đo camera/âm thanh. Hiện chưa có bằng chứng đạt
các mục tiêu nghiệm thu, chưa huấn luyện lại trên dữ liệu iPhone mới.
