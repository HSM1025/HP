import cv2
from Camera import Camera
from IntrusionDetector import IntrusionDetector

camera = Camera('dummy.mp4')
intrusion = IntrusionDetector(camera)

while True:
    frame = camera.capture_frame()
    if frame is None:
        break

    # 움직임 감지 (True/False)
    motion_detected = intrusion.detect_motion(frame)

    print(motion_detected)
    # 프레임 출력
    cv2.imshow("Frame", frame)

    if cv2.waitKey(17) == 27:  # ESC
        break

cv2.destroyAllWindows()
