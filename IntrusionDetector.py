import cv2
from Event import Event

class IntrusionDetector:
    def __init__(self, cameras: list ,ai_analyzer):
        self.cameras = cameras  # 카메라 리스트
        self.motion_caps = [[] for _ in cameras]  # 각 카메라별 motion_cap
        self.intrusion_flags = [False] * len(cameras)  # 각 카메라별 결과 플래그
        self.__aianalyzer = ai_analyzer
        self.callcount0 = 0
        self.callcount1 = 0
        self.callcount2 = 0
        self.fgbg_list = [
            cv2.createBackgroundSubtractorMOG2(
                history=40,
                varThreshold=16,
                detectShadows=False
            ) ,cv2.createBackgroundSubtractorMOG2(
                history=40,
                varThreshold=16,
                detectShadows=False
            ) ,cv2.createBackgroundSubtractorMOG2(
                history=40,
                varThreshold=16,
                detectShadows=False
            )
        ]

        while True:
            frames = []

            # 1) 모든 카메라에서 한 프레임씩 가져오기
            for cam in self.cameras:
                frame = cam.capture_frame()
                if frame is None:
                    frames.append(None)
                else:
                    frames.append(frame)

            if all(f is None for f in frames):
                break

            # 2) 가져온 프레임을 각각 모션 감지
            for idx, frame in enumerate(frames):
                if frame is None:
                    continue

                self.detect_motion(idx, frame)

                if len(self.motion_caps[idx]) > 10:
                    # 모션이 찾아진 카메라 배열
                    self.intrusion_flags[idx] = True
                    if idx==0 and self.callcount0==0:
                        aianalyze = __aianalyzer.analyze(frame,"INTRUDER")
                        if aianalyze:
                            create_Intrusion_Event("Intrusion", "camera1")
                            self.callcount0 += 1

                    elif idx==1 and self.callcount1==0:
                        aianalyze = __aianalyzer.analyze(frame, "INTRUDER")
                        if aianalyze:
                            create_Intrusion_Event("Intrusion", "camera2")
                            self.callcount1 += 1

                    elif idx==2 and self.callcount2==0:
                        aianalyze = __aianalyzer.analyze(frame, "INTRUDER")
                        if aianalyze:
                            create_Intrusion_Event("Intrusion", "camera3")
                            self.callcount2 += 1

            if cv2.waitKey(17) == 27:
                break

        cv2.destroyAllWindows()
    def create_Intrusion_Event(self,type,location):
        EventManager.instance().add_event(Event(type, location))

    def detect_motion(self, idx, frame):
        fgmask = self.fgbg_list[idx].apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) >= 300:
                self.motion_caps[idx].append(True)
                return True

        return False
