from Camera import Camera
from IntrusionDetector import IntrusionDetector

camera1 = Camera('dummy.mp4')
camera2 = Camera('dummy2.mp4')
camera3 = Camera('dummy3.mp4')
camera_List = [camera1, camera2, camera3]
intrusion = IntrusionDetector(camera_List, )
print("결과값 : ", intrusion.motion_caps)
# 객체 생성 후 바로 실행
