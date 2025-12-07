import cv2
from cv2 import bitwise_or
import numpy as np

from EventManager import EventManager
from Event import Event

'''
    화재 감지 클래스
    camera: 화재 감지할 카메라(영상)
    aiAnalyzer: 화재 여부 재검증 시 사용할 AI분석기
    fireDetectedDuration: 화재 감지가 지속된 시간
    fireDurationThreshold: 화재로 판단하는 감지 지속 시간
'''
class FireDetector:
    def __init__(self, camera, ai_analyzer, threshold):
        self.__camera = camera
        self.__aiAnalyzer = ai_analyzer
        self.__fireDetectedDuration = [[] for i in range(len(self.__camera))]
        self.__fireDetectedThreshold = threshold

        detected = [False for i in range(len(self.__camera))] # 각 카메라 화재 감지 여부
        finish = False # 영상 종료 여부
        while True:
            for i in range(len(self.__camera)):
                if detected[i]: # 해당 카메라 화재 감지됐을 시 pass
                    continue

                frame = self.__camera[i].capture_frame()

                # 17ms간 키 입력을 대기 -> 입력 없을 경우 if문 분기 안됨
                # 17ms간 키 입력을 대기 -> Esc키(27)를 눌렀을 경우 break
                # 프레임이 끝났을 경우(읽기 실패) break
                # 60FPS -> 16.67ms ->  약 17ms
                # 30FPS -> 33.3ms -> 약 33ms
                if cv2.waitKey(6) == 27 or frame is None:
                    finish = True # 영상 종료 여부 true
                    break

                if self.analyze_color_pattern(i, frame):
                    if self.__aiAnalyzer.analyze(frame, "Fire"):
                        detected[i] = True # 해당 카메라 화재 감지 여부 true
                        # Fire 이벤트 생성
                        self.create_fire_event("Fire", camera[i].get_location())

                cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Frame", 1133, 640)
                cv2.imshow("Frame", frame)

            if finish: # 영상 종료 시 감지 종료
                break

    def analyze_color_pattern(self, camera_index, frame):
        # 이미지 색 공간 변환
        # cv2.cvtColor(src, code, dst=None, dstCn=0) -> dst
            # src: 변환할 이미지
            # code: 변환하려는 색 공간 지정(flag) ex) cv2.COLOR_BGR2GRAY
            # dst: 출력 이미지(src와 동일한 크기)
            # dstCn: 출력 이미지의 채널 수 지정(0 -> code 채널 수로 지정)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 특정 색상 영역 추출
        # cv2.inRange(src, lowerb, upperb, dst=None) -> dst
            # src: 추출할 이미지
            # lowerb: 영역 중 가장 최소 배열 or 스칼라
            # upperb: 영역 중 가장 최대 배열 or 스칼라
            # dst: 출력 이진 이미지(src와 CV_8U(8비트 부호없는 정수) 유형과 동일한 크기)
        # 빨간색 H: 약 160 ~ 179 / 0 ~ 10, 주황색~노랑색 H: 약 10 ~ 35
        mask1 = cv2.inRange(frame_hsv, (0, 120, 180), (35, 255, 255))
        mask2 = cv2.inRange(frame_hsv, (170, 150, 180), (179, 255, 255))
        color_mask = bitwise_or(mask1, mask2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Duration에 아무것도 없다면 초기 프레임 추가
        if len(self.__fireDetectedDuration[camera_index]) == 0:
            self.__fireDetectedDuration[camera_index] = [gray]
            return False

        # 이미지 차분
        # cv2.absdiff(src1, src2, dst=None) -> dst
            # src1: 차분할 이미지 1
            # src2: 차분할 이미지 2
            # dst: src1 - src2한 이미지
        diff = cv2.absdiff(self.__fireDetectedDuration[camera_index][-1], gray)

        # 불 색깔이고 픽셀의 변화가 20 이상인 경우의 마스크
        mask = (color_mask == 255) & (diff < 20)
        color_mask[mask] = 0

        # return color_mask

        # 전체 frame 크기 연산
        h_img, w_img = frame.shape[:2]
        frame_area = h_img * w_img

        # mask된 frame 크기 연산
        fire_area = np.count_nonzero(color_mask)

        # mask된 frame의 비율 연산
        fire_ratio = fire_area / frame_area
        print('fire_ratio= ' + str(float(fire_ratio)))

        # 전체 frame 중 0.01%가 mask되어있을 경우 불 의심
        if fire_ratio >= 0.0001:
            self.__fireDetectedDuration[camera_index].append(gray)
        else:
            self.__fireDetectedDuration[camera_index].clear()

        # 불을 의심한 프레임이 임계치를 넘을 경우 true 반환 아니라면 false 반환
        if len(self.__fireDetectedDuration[camera_index]) >= self.__fireDetectedThreshold + 1:
            return True
        else:
            return False

    def create_fire_event(self, type, location):
        EventManager.instance().add_event(Event(type, location))