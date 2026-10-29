# Guide — nhận biết chướng ngại vật trên laptop và web

Pipeline dùng chung cho desktop và web local: webcam, DroidCam, luồng RTSP/MJPEG hoặc video đã ghi. Mặc định dùng mô hình ROD tại `models/results/best.pt` nếu có.

## Chạy desktop

Tại thư mục dự án, dùng môi trường Python đã chạy nhận diện được:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Chọn **Webcam / DroidCam**, nhập chỉ số camera (thường 0 hoặc 1), hoặc bấm **Quét camera**. Chọn kích thước suy luận **480** để bắt đầu; so sánh 320/480/640 bằng benchmark phía dưới. Chọn **Video đã ghi** để dùng tệp có sẵn, hoặc **Camera IP** để nhập URL video RTSP/MJPEG. Có thể chọn mô hình `.pt` hoặc `.onnx`; với ONNX dùng mô hình detection có metadata YOLO, dynamic shape nếu cần đổi kích thước.

**Thử: Dừng lại** phát clip tiếng Việt mà không cần mở camera. Tắt **Bật âm thanh trên laptop** để quan sát yên lặng.

PyTorch được nạp trước WinRT để tránh xung đột nạp DLL trên Windows. Nếu một môi trường khác báo c10.dll, dùng lại Python đã chạy được thay vì tự đổi sang môi trường đó.

## Kết nối iPhone bằng DroidCam

1. Cài DroidCam trên iPhone và [DroidCam Client trên Windows](https://droidcam.app/windows/).
2. Mở app iPhone, cấp quyền camera/mạng local, cho iPhone và laptop vào cùng Wi-Fi.
3. Trong client Windows, thêm điện thoại và xác nhận xem được hình. Ban đầu chỉ bật video.
4. Trong Guide chọn **Webcam / DroidCam**, quét camera và thử chỉ số webcam ảo. Guide hiện hiển thị chỉ số, chưa tự gắn tên thiết bị.
5. Nếu Wi-Fi trễ, thử USB theo [hướng dẫn DroidCam](https://droidcam.app/help/): cần cáp dữ liệu, Trust máy tính và Apple Mobile Device Support.
6. Đóng Windows Camera/Zoom nếu chúng đang chiếm camera. Sau khi ngắt kết nối, dừng phiên, kết nối lại client rồi mở phiên mới.

DroidCam là webcam ảo, không cần nhập URL vào Guide. Ô Camera IP dành cho app khác thực sự cung cấp RTSP/MJPEG. Webcam ảo/IP chỉ đưa ảnh RGB, không đưa LiDAR.

## Giọng Việt local và cache

### Chọn giọng cảnh báo khác

Đã có công cụ cài bộ giọng riêng và giữ bộ Microsoft An để đổi lại. Hai mẫu nghe thử nằm trong `assets/audio/previews/khanh_vy.wav` và `assets/audio/previews/hoang_nam.wav`.

```powershell
# Tạo và chọn Khánh Vy. Đổi khanh_vy thành hoang_nam nếu thích giọng nam.
.\.venv-neural\Scripts\python.exe tools/setup_voice.py khanh_vy --prepare --activate

# Đóng app đang chạy, sau đó mở lại. Không cần neural TTS trong luồng cảnh báo.
python main.py

# Xem bộ giọng đã chuẩn bị / bộ đang chọn
python tools/setup_voice.py --list

# Chọn giọng đã chuẩn bị mà không tạo lại âm thanh
python tools/setup_voice.py khanh_vy --activate

# Đổi về giọng Microsoft An
python tools/setup_voice.py an --activate
```

Quá trình tạo giọng neural mất thời gian một lần; chỉ kích hoạt sau khi tất cả đoạn WAV đã hợp lệ. Giọng được ghi trong `voice_settings.json` và được cả desktop/web đọc khi khởi động. Bộ mới nằm trong `assets/audio/voices/<id>/`, không ghi đè `assets/audio/fast/`. Tùy chọn `--tts` vẫn chỉ áp dụng cho câu tự do ngoài ngân hàng đã chuẩn bị, không tự thay giọng cảnh báo nhanh.

Âm thanh được ghép theo thứ tự **lệnh di chuyển → vật cản → hướng → gần/vừa/xa**. Ví dụ: “Đi chậm lại. Người, bên trái, gần.” Khi mask lối đi ổn định xác nhận hướng phía trước, đọc “Đi thẳng, chú ý quan sát” trước thông tin vật cản. Nếu phía trước bị chắn và có lối bên trái/phải ổn định, đọc “Dừng lại. Rẽ trái/phải sau khi kiểm tra lối đi”, rồi mô tả vật cản. Cảnh báo khẩn vẫn bắt đầu bằng “Dừng lại”. Không suy ra lệnh rẽ hoặc đi thẳng chỉ từ việc không thấy vật thể. Các đoạn lệnh và tên vật cản được nạp sẵn vào RAM, không chờ tổng hợp TTS cho các cảnh báo đã chuẩn bị.

Thiết bị phát PCM Windows được mở và làm nóng một lần khi khởi động. Cảnh báo đang phát được đọc hết; trong lúc đó ứng dụng chỉ giữ cảnh báo mới nhất, thay các cảnh báo đang chờ cũ hơn. Âm thanh không bị chồng lên nhau. Stop/Tắt âm thanh vẫn ngắt phát ngay. Những câu không nằm trong ngân hàng vẫn được tổng hợp ở nền và cache trong `assets/audio/<engine>/`. Tùy chọn `--tts` áp dụng cho các câu động này; cảnh báo nhanh dùng bộ từ đã chuẩn bị.

Bộ giọng gốc được tạo bằng Microsoft An; giọng đang chọn được lưu trong `voice_settings.json` và hiện trong trạng thái âm thanh. Những WAV đã chuẩn bị chạy offline kể cả khi không cần nạp TTS. Có thể tạo lại bằng giọng khác hoặc bổ sung tên lớp mới; WAV đầu vào cần PCM 16-bit. Cần khởi động lại app sau khi đổi giọng. Bộ giọng tùy chọn thiếu đoạn phải được bổ sung bằng `tools/setup_voice.py --prepare` trước khi dùng.

```powershell
python tools/prepare_fast_alerts.py
python tools/benchmark_audio_latency.py --samples 10
```

Kết quả mới: `reports/audio_latency_improved.json`; p95 yêu cầu → gửi PCM khoảng 1.4 ms qua 10 mẫu im lặng. Đã bỏ khoảng 120 ms im lặng đầu và 750–780 ms cuối các clip cũ. Số đo này chưa bao gồm độ trễ vật lý của loa/tai nghe; Bluetooth có thể thêm độ trễ riêng.

```powershell
# Chuẩn bị clip và đo tổng thời gian sinh WAV, không phát ra loa
python tools/benchmark_tts.py --engine windows --repeat 3
```

Đã thêm adapter cho [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) và [KorvaTTS](https://github.com/dogenthq/KorvaTTS). Mô hình neural cần tải trọng số lần đầu, sau đó dùng cache local. Chạy trong môi trường riêng để tránh đổi gói của Python đang hoạt động:

```powershell
python -m venv --system-site-packages .venv-neural
.\.venv-neural\Scripts\python.exe -m pip install -r requirements-tts-vieneu.txt -r requirements-tts-korva.txt
.\.venv-neural\Scripts\python.exe tools/benchmark_tts.py --engine vieneu --repeat 3
.\.venv-neural\Scripts\python.exe tools/benchmark_tts.py --engine korva --repeat 3
.\.venv-neural\Scripts\python.exe main.py --tts vieneu
# Hoặc --tts korva
```

VieNeu mặc định ONNX CPU int8, 4 luồng. Nếu máy phát âm méo, đặt `$env:GUIDE_VIENEU_PRECISION="fp32"` rồi đo lại. Chọn giọng bằng `GUIDE_VIENEU_VOICE` (mặc định Mai Anh) hoặc `GUIDE_KORVA_VOICE` (mặc định khanh_vy). Sau khi đổi giọng/precision, xóa hoặc đổi tên cache audio tương ứng để sinh lại.

## Chạy web local

```powershell
python web_app.py
```

Mở **http://127.0.0.1:8000**. Giao diện dùng chung mô hình, cảnh báo và chỉ số với desktop. Chạy một giao diện mỗi lần để tránh hai tiến trình cùng chiếm camera.

Để mở giao diện trên iPhone cùng mạng:

```powershell
python web_app.py --host 0.0.0.0 --port 8000
```

Dùng địa chỉ IP LAN của laptop, ví dụ `http://192.168.1.10:8000`; cho phép Python trên mạng Private nếu Windows hỏi. Âm thanh vẫn phát ở laptop. Web hiện nhận video từ backend/DroidCam, preview MJPEG tối đa 30 FPS và trạng thái cập nhật 4 lần/giây; FPS nhận diện đo riêng. Chưa lấy camera trực tiếp trong Safari, chưa có WebRTC hay HTTPS. Muốn phát triển đường camera Safari cần bổ sung cả WebRTC và HTTPS; không dùng URL HTTP LAN để yêu cầu camera trình duyệt. Server hiện dành cho mạng local tin cậy, không mở ra Internet.

## Đo tốc độ và kiểm thử

Kết quả đo nằm trong [reports/UPGRADE_RESULTS.md](reports/UPGRADE_RESULTS.md). Runtime đạt khoảng 8.5 FPS AI trên video thử, p95 decode→kết quả khoảng 162 ms; camera đọc khoảng 30 FPS. Chưa đạt 45 FPS. Hai TTS neural tạo WAV chậm hơn Microsoft An, nên mặc định giữ Microsoft An + clip cache.

```powershell
python tools/benchmark_runtime.py --source data/videos/train4.mp4 --samples 60 --sizes 320 480 640 --threads 8
python -m unittest discover -s tests -v
```

Benchmark loại 3 lượt làm nóng, đo cùng đoạn video, gồm suy luận và vẽ chú thích; ghi tốc độ phân tích, p50/p95 và thời gian decode vào `reports/runtime_benchmark.json`. Nó không bao gồm thời gian truyền camera, hiển thị hoặc âm thanh, và không đo độ chính xác nhận diện. Có thể lặp `--model` để so PT/ONNX đã xuất. Cấu hình runtime dùng 8 luồng PyTorch CPU theo phép so 1/4/8/12 luồng; chỉnh `GUIDE_TORCH_THREADS` sau khi đo trên máy đích.

Để xuất và so ONNX trong môi trường riêng (đã đo chậm hơn PT trên máy hiện tại):

```powershell
.\.venv-neural\Scripts\python.exe -m pip install "onnx>=1.17,<2"
.\.venv-neural\Scripts\python.exe tools/export_detector.py
.\.venv-neural\Scripts\python.exe tools/benchmark_runtime.py --source data/videos/train4.mp4 --model models/results/best.pt --model models/results/best.onnx
```

Mỗi phiên desktop/web lưu `reports/session_*.json` khi kết thúc:

- FPS camera: tốc độ khung đã decode tới laptop; không phải FPS cảm biến.
- FPS AI: số kết quả suy luận hoàn thành mỗi giây.
- Khung bỏ qua: khung bị thay bằng khung mới hơn khi AI còn bận.
- p50/p95 nhận diện: từ lúc decode xong khung tới lúc hoàn thành nhận diện/vẽ.
- p50/p95 gửi âm thanh: tới lúc gọi playback; chưa đo thời điểm loa thật phát ra tiếng.
- Phân vị dùng cửa sổ tối đa 2.048 mẫu; FPS dùng 2 giây gần nhất. Bỏ qua thời gian truyền Wi-Fi và buffer nội bộ driver/decoder.

Mục tiêu 45 FPS là mục tiêu benchmark phụ thuộc phần cứng, chưa phải khả năng đã đạt. Cần thử thêm webcam/iPhone thật và chạy liên tục 30 phút để đánh giá mất camera, nhiệt độ và cảnh báo muộn.

## Khoảng cách, lối đi và phạm vi chỉ dẫn

- Gần/vừa/xa chỉ dựa kích thước hộp và vị trí trên ảnh; không hiển thị số mét.
- Vật thể gần giao với vùng trước mặt kích hoạt “Dừng lại”; vật thể bên trái/phải tạo cảnh báo đúng phía.
- Hình quá tối/mờ hoặc phát hiện nguy hiểm không đủ tin cậy tạo “Chưa nhìn rõ”.
- Chỉ mask lớp `walkway` đủ tin cậy, liên thông và không bị vật thể che mới cung cấp gợi ý lối đi; cần ổn định ít nhất 1 giây. Mask `road` và hộp detection không xác nhận đường trống.
- Với ROD detection hiện tại, ứng dụng phát cảnh báo vật cản, chưa tự xác nhận được lối đi bằng mask. Không có chỉ dẫn “lùi lại”.
- Mất khung hình mới sẽ hủy chỉ dẫn cũ; camera lỗi cần kết nối lại rồi bắt đầu phiên mới.
- Các lớp ROD như Bike, Electrical Pole, Dustbin, Guard rail, Traffic Cone, Plant Pot, Electrical Box đã được ánh xạ sang tiếng Việt.
- Đo mét từ iPhone LiDAR cần ứng dụng iOS riêng gửi RGB + depth + confidence đồng bộ. Chưa triển khai trong giai đoạn này.

Chỉ demo trong khu vực an toàn có giám sát. Không có cảnh báo không xác nhận lối đi an toàn; ứng dụng chưa thay thế phương tiện trợ giúp di chuyển.

## Dữ liệu nghiên cứu

- [SANPO](https://github.com/google-research-datasets/sanpo_dataset): dữ liệu góc nhìn người đi bộ ngoài trời, có thông tin ngữ nghĩa và chiều sâu; phù hợp để nghiên cứu đường đi và chướng ngại. Kiểm tra điều khoản CC BY 4.0 và yêu cầu ghi công trước khi phân phối dữ liệu hoặc sản phẩm.
- [SurDis](https://github.com/kuanyewleong/surdis): tập trung vào các điểm gián đoạn bề mặt lối đi như bậc lên/xuống, miệng cống hở và chỗ hụt. Xác minh quyền tải, giấy phép và điều kiện sử dụng với chủ dữ liệu trước khi dùng trong huấn luyện hoặc phát hành.
- Chia train/validation/test theo địa điểm hoặc tuyến đường, không chia ngẫu nhiên từng khung hình từ cùng một đoạn video để tránh rò rỉ cảnh gần giống nhau.
- Webcam là nguồn thử nghiệm tương tác; video ghi sẵn và các tập dữ liệu có giấy phép rõ ràng là nguồn đánh giá khả năng tổng quát hóa.

## Quy trình đánh giá prototype

1. Lập bộ video có và không có vật cản, ghi rõ tuyến/địa điểm, điều kiện sáng và loại nguy hiểm.
2. Tính recall cho các tình huống nguy hiểm đã gán nhãn, số cảnh báo sai trên mỗi 5 phút video đường thông thoáng và độ trễ từ lúc khung hình được lấy đến lúc TTS bắt đầu phát. Giao diện hiển thị số khung hình đã xử lý/bỏ qua, p95 thời gian đến nhận diện, p95 đến lúc bắt đầu giọng nói và số cảnh báo.
3. Mục tiêu nghiên cứu ban đầu: recall ít nhất 90% cho nhóm nguy hiểm đã chọn; không quá 1 cảnh báo sai mỗi 5 phút video thông thoáng; p95 độ trễ không quá 700 ms.
4. Chạy liên tục 30 phút, ghi nhận mất camera, khung hình bỏ qua, cảnh báo trùng và cảnh báo muộn.
5. Chỉ tổ chức thử nghiệm có người sau khi đánh giá video đạt mục tiêu, có sự đồng thuận, người giám sát và gậy/chó dẫn đường như bình thường.

Các ngưỡng độ mờ và vùng gần/vừa/xa trong mã là mặc định ban đầu; cần hiệu chỉnh theo webcam và dữ liệu kiểm thử. Đạt các mục tiêu prototype không chứng minh ứng dụng an toàn cho việc tự đi bộ.

## Công cụ baseline

- [data/README.md](data/README.md) hướng dẫn tạo nhãn khoảng thời gian và chạy baseline trên video.
- `tools/label_video.py` gắn nhãn sự kiện để tính recall/cảnh báo sai; các nhãn này không phải bounding box để huấn luyện.
- `tools/evaluate_baseline.py` dùng cùng bộ lọc phát hiện của app, xuất báo cáo JSON vào `reports/`.

## Thu thập dữ liệu và fine-tune mô hình

Ứng dụng hiện có quy trình gán nhãn đa giác trên khung hình video và fine-tune mô hình YOLO segmentation:

1. Quay video từ góc camera dự kiến (ngang ngực/đầu), gồm lối đi trống, người và phương tiện, chướng ngại cố định, lề/bậc và hố/cống. Thu video ở nơi an toàn; không yêu cầu người đang đi phải cầm laptop.
2. Gán nhãn bằng `tools/annotate_training_video.py`. Mỗi phiên quay và địa điểm cần có mã riêng; một địa điểm chỉ được gán vào một tập dữ liệu.
3. Chạy `tools/train_obstacle_model.py`; công cụ kiểm tra manifest, nhãn và rò rỉ địa điểm trước khi gọi Ultralytics.
4. Mô hình tốt nhất được lưu ở `models/walkway_obstacles.pt`. Mở ứng dụng, chọn tệp này trong ô mô hình rồi chạy trên webcam hoặc video.

Mười hai lớp theo thứ tự ID: `person`, `bicycle`, `motorcycle`, `car`, `bus_truck`, `dog`, `pole_tree`, `barrier`, `walkway`, `road`, `curb_step`, `pothole_drain`. Vẽ đa giác quanh vùng hiển thị của từng lớp. `walkway` và `road` cung cấp ngữ cảnh thị giác và không tự phát cảnh báo; các lớp chướng ngại mới dùng cho cảnh báo.

PAVE trên Hugging Face có thể tham khảo cách mô tả khả năng đi lại và ngữ nghĩa cảnh; các chú giải của PAVE không thay thế ảnh SANPO và nhãn phân đoạn tương ứng trong dữ liệu train. Không đưa VLM vào luồng cảnh báo MVP. SANPO có góc nhìn egocentric, ảnh và nhãn panoptic cho một phần khung hình; trang chính thức trỏ tới giấy phép CC BY 4.0, cần ghi công khi sử dụng. SurDis công bố giấy phép CC BY-NC-SA 4.0 và bộ dữ liệu hiện được lưu ở đường dẫn tạm thời, nên không dùng trong sản phẩm thương mại khi chưa có quyền phù hợp. [SANPO](https://github.com/google-research-datasets/sanpo_dataset), [SurDis](https://github.com/kuanyewleong/surdis), [PAVE](https://huggingface.co/datasets/rafiibnsultan/PAVE).

Chi tiết phím tắt, cấu trúc tập dữ liệu, lệnh chạy và lưu ý giấy phép nằm trong [data/README.md](data/README.md). Mô hình ROD đã huấn luyện được đặt ở `models/results/best.pt`; đợt nâng cấp này tập trung vào runtime và giao diện, chưa huấn luyện thêm dataset.

## SegFormer Sidewalk trên Kaggle

Import [kaggle_train_sidewalk_segformer_low_memory.ipynb](kaggle_train_sidewalk_segformer_low_memory.ipynb) vào notebook Kaggle mới, bật GPU/Internet và Secret `HF_TOKEN` của tài khoản có quyền đọc `segments/sidewalk-semantic`. Notebook có đầy đủ code; không cần tải thêm script Python lên Kaggle.

Lần đầu giữ `RUN_MODE="smoke"` để kiểm tra 20 optimizer steps, validation, lưu/tải checkpoint và export. Khi thấy `SMOKE_COMPLETED`, đổi `RUN_MODE="full"`, restart kernel/session rồi chạy lại từ đầu để huấn luyện 10 epoch. Full dùng cùng cách chia ngẫu nhiên 800/100/100 của notebook nguồn; đây là baseline nội bộ, cần thêm tập video theo tuyến độc lập để đánh giá ứng dụng thực tế.

Bản này dùng một GPU, ảnh 384, batch 1 và gradient accumulation 4; thu nhỏ ảnh/mask từng ảnh ra ổ đĩa, tính IoU từ ma trận đếm từng batch, ghi RAM/VRAM vào `memory_log.jsonl`. Dùng Transformers 4.57.1 cố định và giữ PyTorch có sẵn của Kaggle. File mô hình full nằm ở `/kaggle/working/segformer_sidewalk_best.zip`; checkpoint để resume nằm trong `segformer_sidewalk_full_checkpoints/`. `TRAINING_COMPLETED` xác nhận đã hoàn tất cả train, test và export.

Nguồn được duy trì tại [tools/train_sidewalk_segformer_kaggle.py](tools/train_sidewalk_segformer_kaggle.py); [tools/build_sidewalk_notebook.py](tools/build_sidewalk_notebook.py) dựng lại notebook độc lập từ script này. Đầu ra SegFormer là thư mục Hugging Face với `model.safetensors`, không tự thay detector ROD của ứng dụng.
