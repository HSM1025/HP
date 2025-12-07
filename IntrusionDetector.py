import cv2
from Camera import Camera
from Event import Event

#모션 감지 캐퍼시터
motion_cap = []

class IntrusionDetector:
    def __init__(self, camera):
        self.camera = camera
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=40,  #배경 모델 학습할 프레임수를 40으로 정함
            varThreshold=16, #픽셀 색상 변경 허용범위
            detectShadows=False #그림자 감지 여부
        )

    #감지 및 카운트
    def run(self):
        while True:
            frame = self.camera.capture_frame()
            if frame is None:
                break

            # 움직임 감지
            motion_detected = self.detect_motion(frame)
            if motion_detected:
                motion_cap.append(motion_detected)
                #10프레임 이상 감지시 True 리턴
                if len(motion_cap)>10:
                    return True
            print(motion_detected)
            print(motion_cap)

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

        # 움직임 있는지 판정
        for cnt in contours:
            if cv2.contourArea(cnt) >= 300:  # 일정 크기 이상 움직임이면 True
                return True
        return False  # 아무것도 없으면 False