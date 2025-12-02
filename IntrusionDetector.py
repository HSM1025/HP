import cv2

class IntrusionDetector:
    def __init__(self, camera):
        self.camera = camera
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=20,
            varThreshold=16,
            detectShadows=False
        )

    def detect_motion(self, frame):
        # 배경 차분 마스크 만들기
        fgmask = self.fgbg.apply(frame)

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motions = []
        for cnt in contours:
            # 너무 작은 움직임은 무시
            if cv2.contourArea(cnt) < 300:
                continue

            # 윤곽선을 감싸는 박스 좌표 저장
            x, y, w, h = cv2.boundingRect(cnt)
            motions.append((x, y, w, h))

        return motions, fgmask
