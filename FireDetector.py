import cv2
import numpy as np
from Event import Event

'''
    화재 감지 클래스
    camera: 화재 감지할 카메라(영상)
    aiAnalyzer: 화재 여부 재검증 시 사용할 AI분석기
    fireDetectedDuration: 화재 감지가 지속된 시간
    fireDurationThreshold: 화재로 판단하는 감지 지속 시간
'''
class FireDetector:
    def __init__(self, camera, ai_analyzer, duration, threshold):
        self.__camera = camera
        self.__aiAnalyzer = ai_analyzer
        self.__fireDetectedDuration = duration
        self.__fireDetectedThreshold = threshold

    def analyze_color_pattern(self, frame):
        # 이미지 색 공간 변환
        # cv2.cvtColor(src, code, dst=None, dstCn=0) -> dst
            # src: 변환할 이미지
            # code: 변환하려는 색 공간 지정(flag) ex) cv2.COLOR_BGR2GRAY
            # dst: 출력 이미지(src와 동일한 크기)
            # dstCn: 출력 이미지의 채널 수 지정(0 -> code 채널 수로 지정)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Duration에 아무것도 없다면 초기 프레임 추가 -> 메서드 종료
        if len(self.__fireDetectedDuration) == 0:
            self.__fireDetectedDuration.append(frame_hsv)
            return False

        # 특정 색상 영역 추출
        # cv2.inRange(src, lowerb, upperb, dst=None) -> dst
            # src: 추출할 이미지
            # lowerb: 영역 중 가장 최소 배열 or 스칼라
            # upperb: 영역 중 가장 최대 배열 or 스칼라
            # dst: 출력 이진 이미지(src와 CV_8U(8비트 부호없는 정수) 유형과 동일한 크기)
        # 빨간색 H: 약 160 ~ 179 / 0 ~ 10, 주황색~노랑색 H: 약 10 ~ 25
        mask1 = cv2.inRange(frame_hsv, (0, 100, 150), (25, 255, 255))
        mask2 = cv2.inRange(frame_hsv, (160, 100, 150), (179, 255, 255))
        mask = mask1 | mask2

        # 이전 frame V값 추출
        pre_frame_v = self.__fireDetectedDuration[len(self.__fireDetectedDuration) - 1][:, :, 2]
        # 현재 frame V값 추출
        frame_v = frame_hsv[:, :, 2]

        # 마스킹된 부분의 이전 frame V값 표준편차
        pre_std_v =  np.std(pre_frame_v[mask > 0])
        # 마스킹된 부분의 현재 frame V값 표준편차
        std_v = np.std(frame_v[mask > 0])

        # 이전frame과 현재 frame V값 표준편차 차이 계산
        diff = abs(pre_std_v - std_v)
        print('diff= ' + str(float(diff)))

        # 밝은 픽셀 비율(마스크된 V값 중 200(밝음) 이상 / 마스크된 V값) 계산
        ratio = np.sum(frame_v[mask > 0] > 200) / frame_v[mask > 0].size
        print('ratio= ' + str(float(ratio)))

        # 이전frame과 현재 frame V값의 차이가 5이상이고 밝은 픽셀 비율이 0.03이상일 경우 불로 감지
        if diff >= 5 and ratio >= 0.03:
            self.__fireDetectedDuration.append(frame_hsv)
        else:
            self.__fireDetectedDuration.clear()

        # 불을 감지한 프레임이 임계치를 넘을 경우 true 반환 아니라면 false 반환
        if len(self.__fireDetectedDuration) >= self.__fireDetectedThreshold:
            return True
        else:
            return False

    def create_fire_event(self, type, location):
        return Event(type, location)