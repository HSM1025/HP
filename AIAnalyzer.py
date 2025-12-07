import cv2
import numpy as np
import tensorflow as tf
import os

# ★ 우리가 만든 전처리 도구들 가져오기
from FirePreprocessing import extract_fire_roi_and_pad
from IntrusionPreprocessing import extract_intruder_roi_and_pad


class AIAnalyzer:
    def __init__(self, confidence_threshold=0.5):
        """
        초기화: 화재 및 침입자 감지용 AI 모델을 로드합니다.
        """
        self.confidence_threshold = confidence_threshold
        self.fire_model = None
        self.intruder_model = None

        # 1. 화재 감지 모델 로드 (TensorFlow - CNN)
        # (마지막으로 작업한 128x128 입력 모델)
        try:
            if os.path.exists('fire_classification_model.h5'):
                self.fire_model = tf.keras.models.load_model('fire_classification_model.h5', compile=False)
                print("[AIAnalyzer] 화재 모델 로드 완료 (fire_classification_model.h5)")
            else:
                print("[Warning] 화재 모델 파일이 없습니다.")
        except Exception as e:
            print(f"[Error] 화재 모델 로드 실패: {e}")

        # 2. 침입자 감지 모델 로드 (TensorFlow - MobileNetV2)
        try:
            if os.path.exists('intruder_classification_model.h5'):
                self.intruder_model = tf.keras.models.load_model('intruder_classification_model.h5', compile=False)
                print("[AIAnalyzer] 침입자 모델 로드 완료 (intruder_classification_model.h5)")
            else:
                print("[Warning] 침입자 모델 파일이 없습니다.")
        except Exception as e:
            print(f"[Error] 침입자 모델 로드 실패: {e}")

    def _preprocess_for_fire(self, frame):
        """ 화재 모델 전용 전처리 (128x128, 1/255 정규화) """
        # 1. ROI 추출 및 패딩
        processed_bgr = extract_fire_roi_and_pad(frame, target_size=(128, 128))
        # 2. RGB 변환 및 정규화
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        img_normalized = processed_rgb.astype('float32') / 255.0
        # 3. 배치 차원 추가 (1, 128, 128, 3)
        return np.expand_dims(img_normalized, axis=0)

    def _preprocess_for_intruder(self, frame):
        """ 침입자 모델 전용 전처리 (224x224, MobileNetV2 전처리) """
        # 1. Haar Cascade로 사람 찾기 및 패딩
        processed_bgr = extract_intruder_roi_and_pad(frame, target_size=(224, 224))
        # 2. RGB 변환
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        # 3. MobileNetV2 전용 스케일링 (-1 ~ 1)
        processed_final = tf.keras.applications.mobilenet_v2.preprocess_input(processed_rgb)
        # 4. 배치 차원 추가 (1, 224, 224, 3)
        return np.expand_dims(processed_final, axis=0)

    def analyze(self, frame, mode):
        """
        다이어그램에 명시된 핵심 메서드
        Args:
            frame: 카메라에서 받은 이미지
            mode: "FIRE" 또는 "INTRUDER"
        Returns:
            Boolean (감지되면 True, 아니면 False)
        """
        if frame is None or frame.size == 0:
            return False

        detected = False
        probability = 0.0

        # ===========================
        # 모드 1: 화재 감지 (FIRE)
        # ===========================
        if mode.upper() == "FIRE":
            if self.fire_model is None:
                return False

            # 전처리 -> 예측
            input_data = self._preprocess_for_fire(frame)
            prediction = self.fire_model.predict(input_data, verbose=0)
            score = prediction[0][0]  # Sigmoid 결과 (0~1)

            # [판단 로직]
            # 학습 시 class_mode='binary' 였고, 보통 Fire=0, Non-Fire=1 로 매핑됨
            # 점수가 낮을수록(0에 가까울수록) 화재일 확률이 높음
            # (반대라면 score > self.confidence_threshold 로 수정하세요)
            if score < (1.0 - self.confidence_threshold):
                detected = True
                probability = (1.0 - score) * 100
            else:
                detected = False

        # ===========================
        # 모드 2: 침입자 감지 (INTRUDER)
        # ===========================
        elif mode.upper() == "INTRUDER":
            if self.intruder_model is None:
                return False

            # 전처리 -> 예측
            input_data = self._preprocess_for_intruder(frame)
            prediction = self.intruder_model.predict(input_data, verbose=0)
            score = prediction[0][0]

            # [판단 로직]
            # Intruder=0, Normal=1 이라고 가정 (알파벳순)
            # 점수가 낮을수록 침입자
            if score < (1.0 - self.confidence_threshold):
                detected = True
                probability = (1.0 - score) * 100
            else:
                detected = False

        else:
            print(f"[Error] 알 수 없는 모드입니다: {mode}")
            return False

        # 디버깅용 출력 (필요 없으면 주석 처리)
        if detected:
            print(f"[{mode}] 감지됨! (확신도: {probability:.1f}%)")

        return detected