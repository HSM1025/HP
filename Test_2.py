import cv2
from Camera import Camera
from IntrusionDetector import IntrusionDetector

camera = Camera('dummy6.mp4')
intrusion = IntrusionDetector(camera)

while True:
    frame = camera.capture_frame()
    if frame is None:
        break

    # 움직임 감지
    boxes, mask = intrusion.detect_motion(frame)

    # 박스 시각화
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # 프레임 출력
    cv2.imshow("Frame", frame)
    cv2.imshow("Motion Mask", mask)

    if cv2.waitKey(17) == 27:  # ESC 종료
        break

cv2.destroyAllWindows()