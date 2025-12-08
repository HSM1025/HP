import sys
import cv2
import threading
import queue
import datetime
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

# --- 원본 모듈 임포트 ---
from Camera import Camera
from EventManager import EventManager
from IntrusionDetector import IntrusionDetector
from FireDetector import FireDetector  # ★ FireDetector 임포트 확인
from AIAnalyzer import AIAnalyzer

# --- 1. 전역 변수 설정 ---
video_buffers = {0: None, 1: None, 2: None}
log_queue = queue.Queue()


# --- 2. Camera Wrapper 클래스 ---
class TkCamera(Camera):
    def __init__(self, location, index, update_gui=True):
        super().__init__(location)
        self.index = index
        self.update_gui = update_gui  # GUI 업데이트 여부 결정

    def capture_frame(self):
        frame = super().capture_frame()
        # GUI 업데이트가 켜져 있고 프레임이 정상일 때만 버퍼에 기록
        if frame is not None and self.update_gui:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_buffers[self.index] = rgb_frame
        return frame


# --- 3. EventManager Monkey Patching (로그 연동 핵심) ---
original_add_event = EventManager.add_event


def patched_notify(self, event):
    # 이벤트 발생 시 로그 큐로 메시지 전송
    evt_type = event.get_event_type()
    location = event.get_camera_location()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    msg = f"[{timestamp}] 🚨 {evt_type} 감지됨! 위치: {location}"
    log_queue.put(msg)
    print(msg)  # 콘솔 확인용


def patched_add_event(self, event):
    # 1. 원본 기능 실행 (이벤트 큐에 추가)
    original_add_event(self, event)
    # 2. 강제로 알림(notify) 실행 -> patched_notify 호출됨
    self.notify(event)


# EventManager 기능 교체
EventManager.notify = patched_notify
EventManager.add_event = patched_add_event


# --- 4. 감지 로직 워커 (스레드) ---
def run_intrusion_system(cameras, analyzer):
    try:
        log_queue.put("[System] 침입 감지 시스템 시작...")
        IntrusionDetector(cameras, analyzer)
    except Exception as e:
        log_queue.put(f"[Error] 침입 감지 오류: {e}")


def run_fire_system(cameras, analyzer):
    try:
        log_queue.put("[System] 화재 감지 시스템 시작...")
        # FireDetector(카메라리스트, 분석기, 임계값)
        # 임계값(threshold)은 10~30 정도가 적당함
        FireDetector(cameras, analyzer, 15)
    except Exception as e:
        log_queue.put(f"[Error] 화재 감지 오류: {e}")


# --- 5. Main GUI 클래스 ---
class SecurityApp:
    def __init__(self, root, sources):
        self.root = root
        self.root.title("Smart Security Monitor (Fire & Intrusion)")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2b2b2b")

        self.sources = sources
        self.cameras_intrusion = []
        self.cameras_fire = []
        self.panels = []
        self.photo_images = [None, None, None]

        self.setup_ui()
        self.start_backend()
        self.animate()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 4분할 화면 크기 고정
        main_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        main_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        main_frame.grid_rowconfigure(0, weight=1, uniform="group1")
        main_frame.grid_rowconfigure(1, weight=1, uniform="group1")

        positions = [(0, 0), (0, 1), (1, 0)]
        for i, pos in enumerate(positions):
            lf = tk.LabelFrame(main_frame, text=f" Camera {i + 1} ",
                               font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b",
                               bd=2, relief="groove")
            lf.grid(row=pos[0], column=pos[1], padx=5, pady=5, sticky="nsew")

            lbl = tk.Label(lf, text="Waiting...", bg="black", fg="gray")
            lbl.pack(fill=tk.BOTH, expand=True)
            self.panels.append(lbl)

        # 로그창
        log_lf = tk.LabelFrame(main_frame, text=" System Logs ",
                               font=("Arial", 12, "bold"), fg="#ff5555", bg="#2b2b2b",
                               bd=2, relief="groove")
        log_lf.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.log_text = scrolledtext.ScrolledText(log_lf, state='disabled',
                                                  bg="#1e1e1e", fg="#00ff00",
                                                  font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    def start_backend(self):
        # AI 모델 로드
        try:
            self.analyzer = AIAnalyzer()
        except Exception:
            self.log_message("[Warning] AI 모델 로드 실패")

        # 1. 침입 감지용 카메라 (화면 출력 O)
        for i, src in enumerate(self.sources):
            cam = TkCamera(src, i, update_gui=True)
            self.cameras_intrusion.append(cam)

        # 2. 화재 감지용 카메라 (화면 출력 X - 분석만 수행)
        # 별도의 객체로 만들어야 파일 읽기 충돌이 안 남
        for i, src in enumerate(self.sources):
            cam = TkCamera(src, i, update_gui=False)
            self.cameras_fire.append(cam)

        # 스레드 1: 침입 감지 실행
        t1 = threading.Thread(target=run_intrusion_system,
                              args=(self.cameras_intrusion, self.analyzer),
                              daemon=True)
        t1.start()

        # 스레드 2: 화재 감지 실행
        t2 = threading.Thread(target=run_fire_system,
                              args=(self.cameras_fire, self.analyzer),
                              daemon=True)
        t2.start()

    def log_message(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def animate(self):
        # 로그 업데이트
        while not log_queue.empty():
            try:
                msg = log_queue.get_nowait()
                self.log_message(msg)
            except queue.Empty:
                pass

        # 영상 업데이트
        for i, panel in enumerate(self.panels):
            frame_rgb = video_buffers.get(i)
            if frame_rgb is not None:
                img_pil = Image.fromarray(frame_rgb)

                panel_w = panel.winfo_width()
                panel_h = panel.winfo_height()

                if panel_w > 10 and panel_h > 10:
                    img_w, img_h = img_pil.size
                    scale = min(panel_w / img_w, panel_h / img_h)
                    new_w, new_h = int(img_w * scale), int(img_h * scale)
                    img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

                img_tk = ImageTk.PhotoImage(image=img_pil)
                panel.configure(image=img_tk, text="")
                self.photo_images[i] = img_tk

        self.root.after(30, self.animate)


if __name__ == "__main__":
    # 영상 경로 설정 (테스트용으로 웹캠 0 사용하거나 파일 경로 입력)
    # video_sources = [0, 0, 0]
    video_sources = ["cam1.mp4", "cam2.mp4", "cam3.mp4"]

    root = tk.Tk()
    app = SecurityApp(root, video_sources)
    root.mainloop()