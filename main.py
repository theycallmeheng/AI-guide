"""Desktop frontend for the shared obstacle guide runtime."""
from __future__ import annotations
import argparse
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
import cv2
from PIL import Image, ImageTk
from pipeline import GuidePipeline, PipelineConfig, default_model
from obstacle_detector import open_camera

APP_DIR = Path(__file__).resolve().parent

class ObstacleGuideApp:
    def __init__(self, root, engine=None, model=None):
        self.root = root
        root.title("Guide • Nhận biết chướng ngại vật")
        root.geometry("1200x810")
        root.minsize(1040, 710)
        self.pipeline = GuidePipeline(engine=engine)
        self.source = tk.StringVar(value="Webcam / DroidCam")
        self.camera = tk.StringVar(value="0")
        self.model = tk.StringVar(value=model or str(default_model()))
        self.video = tk.StringVar()
        self.url = tk.StringVar()
        self.imgsz = tk.StringVar(value="480")
        self.device = tk.StringVar(value="auto")
        self.confidence = tk.DoubleVar(value=.30)
        self.sound = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Sẵn sàng")
        self.voice = tk.StringVar(value="Đang chuẩn bị âm thanh…")
        self.guidance = tk.StringVar(value="Chọn camera hoặc video để bắt đầu.")
        self.summary = tk.StringVar()
        self.metric_vars = [tk.StringVar(value="—") for _ in range(4)]
        self._last_frame = -1
        self._photo = None
        self.scanning = False
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self._poll)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background="#edf2f6", foreground="#142b3c")
        style.configure("TFrame", background="#edf2f6")
        style.configure("TLabel", background="#edf2f6")
        style.configure("TButton", padding=(10, 8))
        style.configure("Accent.TButton", background="#087f8c", foreground="white", font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton", background=[("active", "#066b76"), ("disabled", "#c0cbd1")])
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"))
        style.configure("Muted.TLabel", foreground="#516b7d", font=("Segoe UI", 9))
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Guide / Cảnh báo chướng ngại", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Xử lý local • Khoảng cách tương đối • Demo có giám sát", style="Muted.TLabel").pack(anchor="w", pady=(0, 16))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        side = ttk.Frame(body, width=300)
        side.pack(side="left", fill="y", padx=(0, 18))
        side.pack_propagate(False)
        content = ttk.Frame(body)
        content.pack(side="left", fill="both", expand=True)

        def label(text):
            ttk.Label(side, text=text, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(12, 4))
        label("1. Nguồn hình ảnh")
        combo = ttk.Combobox(side, textvariable=self.source, values=("Webcam / DroidCam", "Video đã ghi", "Camera IP"), state="readonly")
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", lambda _: self._source_changed())
        self.camera_row = ttk.Frame(side)
        ttk.Label(self.camera_row, text="Chỉ số").pack(side="left", padx=(0, 6))
        self.camera_combo = ttk.Combobox(self.camera_row, textvariable=self.camera, values=tuple(str(i) for i in range(10)), width=5)
        self.camera_combo.pack(side="left")
        self.scan = ttk.Button(self.camera_row, text="Quét camera", command=self.scan_cameras)
        self.scan.pack(side="right")
        self.video_row = ttk.Frame(side)
        ttk.Entry(self.video_row, textvariable=self.video, width=18).pack(side="left", fill="x", expand=True)
        ttk.Button(self.video_row, text="Chọn…", command=self.choose_video).pack(side="left", padx=(4, 0))
        self.ip_row = ttk.Frame(side)
        ttk.Entry(self.ip_row, textvariable=self.url).pack(fill="x")
        ttk.Label(self.ip_row, text="URL video RTSP hoặc MJPEG, cùng Wi-Fi.", style="Muted.TLabel", wraplength=290).pack(anchor="w")
        self.source_hint = ttk.Label(side, text="", wraplength=295, style="Muted.TLabel")
        self.source_hint.pack(anchor="w", pady=(5, 0))
        ttk.Button(side, text="Hướng dẫn DroidCam ↗", command=lambda: webbrowser.open("https://droidcam.app/windows/")).pack(fill="x", pady=(6, 0))
        label("2. Mô hình và tốc độ")
        row = ttk.Frame(side)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.model, width=17).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Chọn…", command=self.choose_model).pack(side="left", padx=(4, 0))
        row = ttk.Frame(side)
        row.pack(fill="x", pady=7)
        ttk.Label(row, text="Ảnh AI").pack(side="left")
        ttk.Combobox(row, textvariable=self.imgsz, values=("320", "480", "640"), width=5, state="readonly").pack(side="left", padx=5)
        ttk.Label(row, text="Thiết bị").pack(side="left")
        ttk.Combobox(row, textvariable=self.device, values=("auto", "cpu", "0"), width=6, state="readonly").pack(side="left", padx=5)
        self.conf_label = ttk.Label(side)
        self.conf_label.pack(anchor="w")
        ttk.Scale(side, from_=.15, to=.75, variable=self.confidence,
                  command=lambda _: self.conf_label.configure(text=f"Độ tin cậy: {self.confidence.get():.2f}")).pack(fill="x")
        self.conf_label.configure(text="Độ tin cậy: 0.30")
        label("3. Cảnh báo tiếng Việt")
        ttk.Checkbutton(side, text="Bật âm thanh trên laptop", variable=self.sound,
                        command=lambda: self.pipeline.speech.set_enabled(self.sound.get())).pack(anchor="w")
        ttk.Button(side, text="Thử: Dừng lại", command=lambda: self.pipeline.speech.speak("Dừng lại.", priority=3, clip="stop", max_age=10)).pack(fill="x", pady=5)
        ttk.Label(side, textvariable=self.voice, wraplength=290, style="Muted.TLabel").pack(anchor="w")
        self.start_button = ttk.Button(side, text="▶  Bắt đầu nhận diện", style="Accent.TButton", command=self.start)
        self.start_button.pack(fill="x", pady=(18, 5))
        self.stop_button = ttk.Button(side, text="■  Dừng", command=self.pipeline.stop)
        self.stop_button.pack(fill="x")
        ttk.Label(side, textvariable=self.status, wraplength=290).pack(anchor="w", pady=10)

        cards = ttk.Frame(content)
        cards.pack(fill="x", pady=(0, 10))
        for title, value in zip(("FPS camera", "FPS nhận diện", "p95 nhận diện", "p95 gửi âm thanh"), self.metric_vars):
            card = tk.Frame(cards, bg="white", padx=12, pady=10)
            card.pack(side="left", expand=True, fill="x", padx=(0, 5))
            tk.Label(card, text=title, bg="white", fg="#516b7d", font=("Segoe UI", 9)).pack(anchor="w")
            tk.Label(card, textvariable=value, bg="white", fg="#087f8c", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        self.preview = tk.Label(content, text="Camera sẽ hiển thị tại đây", bg="#101e2a", fg="#a2b6c6", font=("Segoe UI", 14))
        self.preview.pack(fill="both", expand=True)
        alert = tk.Frame(content, bg="#fff3db", padx=14, pady=12)
        alert.pack(fill="x", pady=(10, 0))
        tk.Label(alert, text="CẢNH BÁO / GỢI Ý", bg="#fff3db", fg="#80540a", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(alert, textvariable=self.guidance, bg="#fff3db", fg="#493004", wraplength=690,
                 justify="left", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(content, textvariable=self.summary, style="Muted.TLabel", wraplength=700).pack(anchor="w", pady=(8, 0))
        ttk.Label(outer, text="Ước lượng từ ảnh không phải số đo mét. Chưa phát hiện vật cản không xác nhận lối đi an toàn.",
                  style="Muted.TLabel").pack(anchor="w", pady=(12, 0))
        self._source_changed()

    def _source_changed(self):
        for row in (self.camera_row, self.video_row, self.ip_row):
            row.pack_forget()
        row = self.video_row if self.source.get() == "Video đã ghi" else self.ip_row if self.source.get() == "Camera IP" else self.camera_row
        row.pack(fill="x", pady=7, before=self.source_hint)
        self.source_hint.configure(text="Mở DroidCam trên iPhone và client Windows, rồi chọn chỉ số webcam ảo. Có thể nhập chỉ số trực tiếp." if row is self.camera_row else "Chọn video cục bộ." if row is self.video_row else "Luồng IP chỉ mang ảnh RGB, không chứa LiDAR.")

    def choose_model(self):
        path = filedialog.askopenfilename(filetypes=(("YOLO / ONNX", "*.pt *.onnx"), ("Mọi tệp", "*.*")))
        if path: self.model.set(path)

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=(("Video", "*.mp4 *.mov *.avi *.mkv"),))
        if path: self.video.set(path)

    def scan_cameras(self):
        if self.scanning or self.pipeline.is_running(): return
        self.scanning = True
        self.scan.configure(state="disabled")
        self.status.set("Đang quét webcam và DroidCam…")
        def scan():
            indices = []
            for index in range(10):
                camera = open_camera(index)
                if camera is not None:
                    camera.release()
                    indices.append(index)
            self._scan_result = indices
        threading.Thread(target=scan, daemon=True).start()

    def start(self):
        try:
            mode = self.source.get()
            source = int(self.camera.get()) if mode == "Webcam / DroidCam" else self.video.get().strip() if mode == "Video đã ghi" else self.url.get().strip()
            config = PipelineConfig(source=source, model=self.model.get().strip(), imgsz=int(self.imgsz.get()),
                                    confidence=self.confidence.get(), live_stream=mode == "Camera IP",
                                    device=None if self.device.get() == "auto" else self.device.get())
            self.pipeline.start(config)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Không thể bắt đầu", str(exc))

    def _poll(self):
        if hasattr(self, "_scan_result"):
            indices = self._scan_result
            del self._scan_result
            self.scanning = False
            self.scan.configure(state="normal")
            self.camera_combo.configure(values=[str(i) for i in indices])
            if indices: self.camera.set(str(indices[0]))
            self.status.set(f"Tìm thấy camera: {indices}" if indices else "Chưa tìm thấy camera. Kiểm tra DroidCam và quyền Camera.")
        state = self.pipeline.snapshot()
        if not self.scanning: self.status.set(state["status"])
        self.voice.set(state["speech"])
        self.guidance.set(state["guidance"])
        stats = state["metrics"]
        for variable, key, suffix in zip(self.metric_vars, ("camera_fps", "inference_fps", "p95_frame_to_inference_ms", "p95_frame_to_audio_submit_ms"), ("", "", " ms", " ms")):
            variable.set(f"{stats[key]:.1f}{suffix}" if key in stats else "—")
        self.summary.set(f"Xử lý {stats.get('processed_frames', 0)} • Bỏ qua {stats.get('skipped_frames', 0)} • "
                         f"p50 nhận diện {stats.get('p50_frame_to_inference_ms', 0):.0f} ms • {stats.get('alert_count', 0)} cảnh báo"
                         + (f" • Gửi âm thanh p95 {stats.get('p95_speech_dispatch_ms',0):.1f} ms" if stats.get('audio_samples') else "")
                         + (" • Đã lưu báo cáo trong reports/" if state["report"] else ""))
        running = self.pipeline.is_running()
        self.start_button.configure(state="disabled" if running or self.scanning else "normal")
        self.scan.configure(state="disabled" if running or self.scanning else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        number, packet = self.pipeline.latest_analysis()
        if packet is not None and number != self._last_frame:
            self._last_frame = number
            image = Image.fromarray(cv2.cvtColor(packet["analysis"].frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((max(640, self.preview.winfo_width()), max(360, self.preview.winfo_height())), Image.Resampling.BILINEAR)
            self._photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self._photo, text="")
        elif packet is None and self._photo is not None:
            self._photo = None
            self.preview.configure(image="", text=state["status"])
        self.root.after(33, self._poll)

    def close(self):
        self.pipeline.close()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts", choices=("windows", "korva", "vieneu"), default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    # Load PyTorch before WinRT to avoid the Windows c10.dll initialization conflict.
    import torch
    root = tk.Tk()
    ObstacleGuideApp(root, args.tts, args.model)
    root.mainloop()

if __name__ == "__main__":
    main()
