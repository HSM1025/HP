import cv2
from Event import Event

class IntrusionDetector:
    def __init__(self, camera):
        self.camera = camera
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=40,
            varThreshold=16,
            detectShadows=False
        )

    def detect_motion(self, frame):
        # 배경 차분
        fgmask = self.fgbg.apply(frame)

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 움직임 있는지 판정
        for cnt in contours:
            if cv2.contourArea(cnt) >= 300:  # 일정 크기 이상 움직임이면 True
                return True
        return False  # 아무것도 없으면 False

    def create_intrusion_event(self,type,loaction):
        return Event(self,type,loaction)