import cv2

'''
    카메라(영상) 클래스
    __location: 영상 주소
    __cap: 영상 프레임(numpy.ndarray)
'''
class Camera:
    def __init__(self, location):
        self.__location = location
        # VideoCapture 객체 생성
        # cv2.VideoCapture(filename, apiPreference=None) -> return
            # filename : 비디오 파일 이름, 비디오 스트림 URL 등
            # apiPreference : 선호하는 카메라 처리 방법
            # return : VideoCapture 객체
        self.__cap = cv2.VideoCapture(self.__location)

    def __del__(self):
        self.__cap.release()

    def get_location(self):
        return self.__location

    def capture_frame(self):
        # VideoCapture open 여부 확인
        # cv2.VideoCapture.isOpened() -> return
            # return : VideoCapture open 여부(open -> true, close -> false)
        if not self.__cap.isOpened():
            print("Cannot open camera")
            exit(1)

        # VideoCapture에서 frame 단위로 읽기
        # cv2.VideoCapture.read(image=None) -> return, image
            # 매개변수 image : 아무 역할 없음(값 무시), c++에서 사용
            # return : 읽기 성공 여부(success -> true, fail -> false)
            # image : 현재 프레임(numpy.ndarray)
        ret, frame = self.__cap.read()
        if not ret:
            return None

        return frame
