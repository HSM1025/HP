import cv2
from Camera import Camera
from IntrusionDetector import IntrusionDetector

camera = Camera('dummy6.mp4')
intrusion = IntrusionDetector(camera)

while True:
    frame = camera.capture_frame()
    if frame is None:
        break

    # 움직임 감지 (True/False)
    motion_detected = intrusion.detect_motion(frame)

    # 텍스트로 화면에 표시
    color = (0, 255, 0) if motion_detected else (0, 0, 255)
    cv2.putText(frame, f"Motion: {motion_detected}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 프레임 출력
    cv2.imshow("Frame", frame)

    if cv2.waitKey(17) == 27:  # ESC
        break

cv2.destroyAllWindows()
