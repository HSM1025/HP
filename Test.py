import cv2
from Camera import Camera
from FireDetector import FireDetector
from EventManager import EventManager

camera = Camera('dummy.mp4')
fireDetector = FireDetector(camera, None, [], 30)
while True:
    frame = camera.capture_frame()
    eventManager = EventManager()
    # 17ms간 키 입력을 대기 -> 입력 없을 경우 if문 분기 안됨
    # 17ms간 키 입력을 대기 -> Esc키(27)를 눌렀을 경우 break
    # 프레임이 끝났을 경우(읽기 실패) break
    # 60FPS -> 16.67ms ->  약 17ms
    # 30FPS -> 33.3ms -> 약 33ms
    if cv2.waitKey(17) == 27 or frame is None:
        break

    if fireDetector.analyze_color_pattern(frame):
        event = fireDetector.create_fire_event("Fire", camera.get_location())
        eventManager.add_event(event)

    if eventManager.peek_event() is not None:
        event = eventManager.get_event()
        eventManager.notify(event)
        exit(0)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1133, 640)
    cv2.imshow("Frame", frame)
