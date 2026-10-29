"""Local web console. Video and speech remain on the laptop; no browser camera capture."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit
import cv2
import numpy as np
from pipeline import APP_DIR, GuidePipeline, PipelineConfig, default_model

def options():
    models = sorted(set([p for pattern in ("*.pt", "models/**/*.pt", "models/**/*.onnx") for p in APP_DIR.glob(pattern)]))
    models = [p for p in models if p.suffix == ".onnx" or p.name in {"best.pt", "walkway_obstacles.pt", "yolo11n.pt", "yolov8n.pt"}]
    videos = sorted(p for p in (APP_DIR/"data/videos").rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
    return {"models": [str(p.relative_to(APP_DIR)) for p in models], "videos": [str(p.relative_to(APP_DIR)) for p in videos],
            "default_model": str(default_model().relative_to(APP_DIR))}

class GuideServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, pipeline):
        super().__init__(address, Handler)
        self.pipeline = pipeline
        self.token = secrets.token_urlsafe(32)
        self.jpeg_lock = threading.Lock()
        self.jpeg_key = None
        self.jpeg = None

    def preview(self):
        frame_id, packet = self.pipeline.latest_analysis()
        key = frame_id if packet is not None else -1
        with self.jpeg_lock:
            if key != self.jpeg_key:
                frame = packet["analysis"].frame if packet else np.full((360, 640, 3), (42, 30, 16), dtype=np.uint8)
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0]*960/frame.shape[1])))
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.jpeg, self.jpeg_key = encoded.tobytes(), key
            return self.jpeg

class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, mime="application/json; charset=utf-8"):
        if isinstance(data, dict): data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        if isinstance(data, str): data = data.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            html = (APP_DIR/"web/index.html").read_text(encoding="utf-8").replace("__TOKEN__", self.server.token)
            return self.reply(200, html, "text/html; charset=utf-8")
        if path in ("/app.css", "/app.js"):
            mime = "text/css" if path.endswith(".css") else "text/javascript"
            return self.reply(200, (APP_DIR/"web"/path[1:]).read_bytes(), mime+"; charset=utf-8")
        if path == "/api/state": return self.reply(200, self.server.pipeline.snapshot())
        if path == "/api/options": return self.reply(200, options())
        if path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    image = self.server.preview()
                    if image:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(image)).encode()+b"\r\n\r\n"+image+b"\r\n")
                    time.sleep(1/30)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        return self.reply(404, {"error": "Không tìm thấy."})

    def do_POST(self):
        token = self.headers.get("X-Guide-Token", "")
        origin = self.headers.get("Origin")
        if not hmac.compare_digest(token, self.server.token) or (origin and urlsplit(origin).netloc != self.headers.get("Host")):
            return self.reply(403, {"error": "Phiên web không hợp lệ. Tải lại trang."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= 8192: raise ValueError("Yêu cầu quá lớn.")
            body = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(body, dict): raise ValueError("Yêu cầu không hợp lệ.")
            pipe = self.server.pipeline
            if self.path == "/api/start":
                choices = options()
                model = body.get("model", "")
                if model not in choices["models"]: raise ValueError("Chọn mô hình trong danh sách.")
                mode = body.get("mode", "camera")
                source = body.get("source", 0)
                if mode == "camera": source = int(source)
                elif mode == "video":
                    if source not in choices["videos"]: raise ValueError("Chọn video trong data/videos.")
                    source = str(APP_DIR/source)
                elif mode != "ip": raise ValueError("Nguồn không hợp lệ.")
                pipe.start(PipelineConfig(source=source, model=str(APP_DIR/model), confidence=float(body.get("confidence", .3)),
                                          imgsz=int(body.get("imgsz", 480)), live_stream=mode == "ip"))
            elif self.path == "/api/stop": pipe.stop()
            elif self.path == "/api/voice": pipe.speech.speak("Dừng lại.", priority=3, clip="stop", max_age=10)
            elif self.path == "/api/sound": pipe.speech.set_enabled(bool(body.get("enabled", True)))
            else: return self.reply(404, {"error": "Không tìm thấy."})
            self.reply(200, {"ok": True})
        except (ValueError, RuntimeError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        # Never log IP-camera URLs or credentials.
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--tts", choices=("windows", "korva", "vieneu"), default=None)
    args = parser.parse_args()
    import torch  # Must initialize before Windows speech.
    pipe = GuidePipeline(engine=args.tts)
    server = GuideServer((args.host, args.port), pipe)
    print(f"Guide web: http://{args.host}:{server.server_port} • Âm thanh phát trên laptop")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        pipe.close()

if __name__ == "__main__": main()
