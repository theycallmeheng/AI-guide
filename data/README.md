# Chuẩn bị video gắn nhãn để đo baseline

## 1. Video đầu vào

- Dùng video đi bộ đã ghi sẵn hoặc cảnh thử nghiệm có người giám sát; không vừa đi vừa mang laptop.
- Mỗi video cần có toàn bộ thời lượng được rà soát. Các khoảng không được gắn nhãn được xem là đã kiểm tra và không có nguy hiểm thuộc danh sách.
- Giữ các video cục bộ; không đưa video có mặt, biển số hoặc thông tin nhận dạng lên Git hay dịch vụ ngoài nếu chưa có quyền và sự đồng thuận.
- Tách tuyến đường/địa điểm giữa train, validation và test. Không chia các đoạn kề nhau của cùng một lần quay sang nhiều tập.

## 2. Gắn nhãn sự kiện cho baseline

Sau khi cài `requirements.txt`, chạy:

```powershell
python tools/label_video.py "data/videos/tuyen_thu.mp4"
```

Trình xem sẽ tạo `data/videos/tuyen_thu_annotations.csv`. Phím tắt:

- `Space`: phát/tạm dừng; nên tạm dừng trước khi đánh dấu.
- `n` / `p`: chọn lớp tiếp theo/trước đó.
- `[` / `]`: đánh dấu đầu/cuối khoảng thời gian nguy hiểm; mỗi sự kiện được lưu ngay.
- `d`: xoay hướng giữa bất kỳ/phía trước/bên trái/bên phải.
- `b`: xoay mức giữa bất kỳ/gần/vừa/xa.
- `j` / `l`: lùi/tiến 1 giây; `J` / `L`: lùi/tiến 5 giây.
- `u`: hủy sự kiện đang mở hoặc xóa nhãn cuối; `s`: lưu; `q`: lưu và thoát.

CSV có các cột `video,start_sec,end_sec,class_name,direction,distance_band`. Đây là nhãn theo khoảng thời gian để đo recall sự kiện và cảnh báo sai; **chưa phải nhãn bounding box/mask để train YOLO**.

## 3. Chạy baseline

```powershell
python tools/evaluate_baseline.py `
  --video "data/videos/tuyen_thu.mp4" `
  --annotations "data/videos/tuyen_thu_annotations.csv" `
  --model "models/yolo11n.pt"
```

Mặc định lấy mẫu mỗi 0,5 giây. Báo cáo JSON trong `reports/` gồm recall theo sự kiện và lớp, số cảnh báo sai mỗi 5 phút, số khung hình chưa rõ và p95 thời gian nhận diện. Chỉ số TTS đo bằng thời gian thực nên xem trong ứng dụng webcam/video.

## 4. Chuẩn bị dữ liệu để train sau baseline

Khi baseline chỉ ra lớp nào bị bỏ sót, cần nhãn bounding box cho mô hình detect hoặc mask cho mô hình segmentation. SANPO có panoptic segmentation và depth nhưng cần chuyển đổi/chọn taxonomy phù hợp. SurDis cung cấp dữ liệu discontinuity bề mặt; giấy phép hiện công bố là CC BY-NC-SA 4.0, nên chỉ dùng cho nghiên cứu phi thương mại nếu chưa được cấp phép khác. Không đưa nhãn khoảng thời gian ở trên vào pipeline YOLO như thể chúng là bounding box.

## 5. Quay video và gán nhãn để fine-tune

Quy trình prototype hiện dùng **đa giác segmentation** trên video quay từ góc camera dự kiến. Lấy mẫu khung hình cách nhau mặc định 1 giây; người gán nhãn có thể bỏ qua cảnh trùng. Gán đa giác quanh vùng của mọi lớp mục tiêu xuất hiện, kể cả vùng không nguy hiểm như mặt đường và lối đi. Đừng vẽ một hộp chữ nhật cho lớp bề mặt lớn nếu ranh giới của nó nhìn thấy được.

### Tách dữ liệu trước khi gán nhãn

- Chia theo **địa điểm/tuyến đường**, rồi gán mỗi phiên quay vào đúng một tập: train, val hoặc test.
- Không cho cùng một địa điểm, phiên quay, hay các đoạn cắt từ cùng một video vào nhiều tập. Công cụ kiểm tra location_id và session_id trong session_manifest.csv, đồng thời từ chối khi phát hiện một mã bị dùng ở nhiều split.
- Ví dụ: dùng route_train_01, route_val_01, route_test_01 làm các mã địa điểm khác nhau; quay nhiều lần ở một tuyến thì giữ tất cả trong cùng split.
- Chỉ ghi video do nhóm có quyền sử dụng; không chia sẻ hình có khuôn mặt/biển số hoặc dữ liệu cá nhân khi chưa xử lý và có sự đồng thuận.

### Lớp nhãn

| ID | Tên lưu trong dataset | Cách đánh dấu |
| --- | --- | --- |
| 0 | person | Người đi bộ |
| 1 | bicycle | Xe đạp |
| 2 | motorcycle | Xe máy |
| 3 | car | Ô tô |
| 4 | bus_truck | Xe buýt hoặc xe tải |
| 5 | dog | Chó |
| 6 | pole_tree | Cột hoặc cây |
| 7 | barrier | Rào chắn/vật chắn |
| 8 | walkway | Vùng lối đi dự kiến |
| 9 | road | Vùng mặt đường xe chạy |
| 10 | curb_step | Lề đường hoặc bậc lên/xuống |
| 11 | pothole_drain | Hố, chỗ hụt hoặc miệng cống |

walkway và road là vùng ngữ cảnh, không tự tạo cảnh báo thoại. Các lớp người/phương tiện và chướng ngại mới được chuyển thành cảnh báo của prototype.

### Gán nhãn video

~~~powershell
python tools/annotate_training_video.py "data/videos/route_train_01.mp4" --session-id "session_train_001" --location-id "route_train_01" --split train
~~~

Phím và thao tác:

- Chuột trái: thêm đỉnh đa giác; chuột phải: khép đa giác (cần tối thiểu 3 đỉnh).
- [ / ]: chuyển lớp; tên lớp và số ID hiện ở đầu cửa sổ.
- u: bỏ đỉnh vừa thêm; x: xóa đa giác vừa khép; c: xóa đa giác đang vẽ.
- s: lưu khung hình và nhãn, kể cả khung hình thông thoáng không có đa giác; n: bỏ qua khung hình; q: kết thúc.

Ảnh và nhãn được lưu vào data/training/images/<split>/<session_id>/ và data/training/labels/<split>/<session_id>/; file session_manifest.csv ghi mã địa điểm, phiên quay và split. PAVE trên Hugging Face có thể tham khảo cách mô tả vùng dễ/khó đi lại, nhưng các chú giải văn bản/nhãn của PAVE không phải ảnh train và không tự ghép với ảnh SANPO. Tập ảnh SANPO cần được tải riêng và chỉ dùng cùng nhãn tương ứng sau khi kiểm tra schema/điều khoản của từng phần dữ liệu. WAD/GuideDog và VLM không nằm trong pipeline detector MVP.

### Huấn luyện và đánh giá

Sau khi cài dependencies theo README gốc và đã có các địa điểm train và validation riêng, chạy:

~~~powershell
python tools/train_obstacle_model.py --epochs 50 --imgsz 640 --batch 4
~~~

Mặc định dùng trọng số segmentation nhỏ yolo11n-seg.pt; Ultralytics chọn CPU/GPU phù hợp nếu không truyền --device. Có thể dùng --device cpu trên máy không có GPU rời hoặc giảm --batch nếu thiếu bộ nhớ. Nếu có ảnh test ở địa điểm giữ riêng, script sẽ chỉ đánh giá tập này sau khi train và không dùng nó để cập nhật trọng số. Mô hình tốt nhất được chép vào models/walkway_obstacles.pt; chọn tệp đó trong ứng dụng.

Không chạy train khi tập train/validation thiếu nhãn đa giác dương tính. Các mục tiêu prototype (recall, cảnh báo sai, độ trễ) vẫn cần đo bằng video kiểm thử riêng; chỉ số segmentation không đồng nghĩa với độ an toàn khi đi bộ.
