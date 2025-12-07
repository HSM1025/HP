import cv2
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
        # 빨간색 H: 약 160 ~ 179 / 0 ~ 10, 주황색~노랑색 H: 약 10 ~ 35
        mask1 = cv2.inRange(frame_hsv, (0, 120, 180), (35, 255, 255))
        mask2 = cv2.inRange(frame_hsv, (170, 150, 200), (179, 255, 255))
        mask = mask1 | mask2

        # 영역 윤곽선 추출
        # cv2.findContours(image, mode, method, contours=None, hierarchy=None, offset=None) -> contours, hierarchy
            # image: 추출할 이미지
            # mode: 윤곽선 추출 모드
                # RETR_EXTERNAL: 영역 바깥쪽 윤곽선 추출
                # RETR_LIST: 영역 모든 윤곽선 추출 (1계층)
                # RETR_CCOMP: 영역 모든 윤곽선 추출 (2계층)
                # RETR_TREE: 영역 모든 윤곽선 추출(모든 계층)
            # method: 영역 근사치 방식
                # CHAIN_APPROX_NONE: 근사 없음
                # CHAIN_APPROX_SIMPLE: 꼭짓점 좌표 제공
            # contours: 검출한 윤곽선 좌표
            # hierarchy: 윤곽선 계층 정보
            # offset: ROI 등으로 인해 이동한 윤곽선 좌표 오프셋
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 전체 frame 크기 연산
        h_img, w_img = frame.shape[:2]
        frame_area = h_img * w_img

        # mask된 frame 크기 연산
        fire_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            fire_area += area

        # mask된 frame의 비율 연산
        fire_ratio = fire_area / frame_area
        print('fire_ratio= ' + str(float(fire_ratio)))

        # 전체 frame 중 4%가 mask되어있을 경우 불 감지
        if fire_ratio >= 0.04:
            self.__fireDetectedDuration.append(frame_hsv)
        else:
            self.__fireDetectedDuration.pop(0)

        # 불을 감지한 프레임이 임계치를 넘을 경우 true 반환 아니라면 false 반환
        if len(self.__fireDetectedDuration) >= self.__fireDetectedThreshold:
            return True
        else:
            return False

    def create_fire_event(self, type, location):
        return Event(type, location)