from Camera import Camera
from FireDetector import FireDetector

camera1 = Camera('dummy_error.mp4')
camera2 = Camera('dummy5.mp4')
camera3 = Camera('dummy.mp4')
FireDetector([camera2], None, 10)
