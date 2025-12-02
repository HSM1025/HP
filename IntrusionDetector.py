import cv2

"""
배경 차분기
history : 학습할 프레임 수 
varThreshold :움직임 감지 임계값
detectShadows : 그림자 탐지 기능
"""
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=50,
    varThreshold=16,
    detectShadows=False
)

class IntrusionDetector:
    def __init__(self, camera, ai_analyzer):
        self.camera = camera
        self.ai_analyzer = ai_analyzer


    def detect_motion(self, frame):
        fgmask = fgbg.apply(frame)
        return fgmask

