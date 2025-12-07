from Camera import Camera
from IntrusionDetector import IntrusionDetector

camera = Camera('dummy.mp4')
intrusion = IntrusionDetector(camera)

# 객체 생성 후 바로 실행
print("결과값 : ",intrusion.intrusion_over_10)
