'''
    이벤트 클래스
    eventType: 감지된 이벤트 타입(Fire/Intrusion)
    cameraLocation: 이벤트 감지된 카메라 영상 주소
'''
class Event:
    def __init__(self, event_type, camera_location):
        self.__eventType = event_type
        self.__cameraLocation = camera_location

    def get_event_type(self):
        return self.__eventType

    def get_camera_location(self):
        return self.__cameraLocation