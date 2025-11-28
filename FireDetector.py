import cv2

'''
    화재 감지 클래스
    camera: 화재 감지할 카메라(영상)
    aiAnalyzer: 화재 여부 재검증 시 사용할 AI분석기
    fireDetectedDuration: 화재 감지가 지속된 시간
    fireDurationThreshold: 화재로 판단하는 감지 지속 시간
'''
class FireDetector:
    def __init__(self, camera, ai_analyzer, duration, threshold):
        self.camera = camera
        self.aiAnalyzer = ai_analyzer
        self.fireDetectedDuration = duration
        self.fireDetectedThreshold = threshold

    def analyzeColorPattern(self, frame):
        # 이미지 색 공간 변환
        # cv2.cvtColor(src, code, dst, dstCn=0) -> dst
            # src: 변환할 이미지
            # code: 변환하려는 색 공간 지정(flag) ex) cv2.COLOR_BGR2GRAY
            # dst: 출력 이미지(선택, src와 동일한 크기)
            # dstCn: 출력 이미지의 채널 수 지정(0 -> code 채널 수로 지정)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 특정 색상 영역 추출
        # cv2.inRange(src, lowerb, upperb, dst) -> dst
            # src: 추출할 이미지
            # lowerb: 영역 중 가장 최소 배열 or 스칼라
            # upperb: 영역 중 가장 최대 배열 or 스칼라
            # dst: 출력 이진 이미지(선택, src와 CV_8U(8비트 부호없는 정수) 유형과 동일한 크기)
        # 빨간색 H: 약 160 ~ 179 / 0 ~ 10, 주황색~노랑색 H: 약 10 ~ 25
        mask1 = cv2.inRange(frame_hsv, (0, 50, 50), (25, 255, 255))
        mask2 = cv2.inRange(frame_hsv, (160, 50, 50), (179, 255, 255))
        mask = mask1 | mask2
        return mask