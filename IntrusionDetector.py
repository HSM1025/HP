import cv2
from Camera import Camera
from Event import Event

class IntrusionDetector:
    def __init__(self, camera):
        self.camera = camera
        self.motion_cap = []
        self.intrusion_over_10 = False
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=40,
            varThreshold=16,
            detectShadows=False
        )

        while True:
            frame = self.camera.capture_frame()
            if frame is None:
                break

            motion_detected = self.detect_motion(frame)
            print(motion_detected)
            if len(self.motion_cap) > 10 :
                self.intrusion_over_10 = True
                break
            # 프레임 출력
            cv2.imshow("Frame", frame)
            if cv2.waitKey(17) == 27:  # ESC
                break

        cv2.destroyAllWindows()

    def detect_motion(self, frame):
        # 배경 차분
        fgmask = self.fgbg.apply(frame)

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(self.motion_cap)
        # 움직임 있는지 판정
        for cnt in contours:
            if cv2.contourArea(cnt) >= 300:
                self.motion_cap.append(True)

        return False
