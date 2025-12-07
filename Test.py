from Camera import Camera
from FireDetector import FireDetector

camera1 = Camera('dummy_error.mp4')
camera2 = Camera('dummy.mp4')
camera3 = Camera('dummy3.mp4')
FireDetector([camera1, camera2, camera3], None, 10)
