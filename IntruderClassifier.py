import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# ★★★ 침입자 전용 전처리 함수 가져오기 ★★★
from IntruderPreprocessing import extract_intruder_roi_and_pad


# --- 모델 구조 (학습 때와 동일) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # [1단계] 특징 추출 (Conv -> Batch Norm -> ReLU -> Pool)
        # 이미지 크기 변화: 128 -> 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 학습 안정화
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # [2단계]
        # 이미지 크기 변화: 64 -> 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # [3단계] (새로 추가됨!) 더 깊게 보기
        # 이미지 크기 변화: 32 -> 16
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # [4단계] 판단 (FC Layer)
        # 최종 크기: 128채널 * 16 * 16
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 50%는 잊어버려서 과적합 방지
            nn.Linear(512, 2)  # 불 vs 안불
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # 일렬로 펴기
        x = self.fc(x)
        return x


class IntruderClassifier:
    def __init__(self, model_path='my_cnn_model_2.pth'):
        """
        초기화: 침입자 감지 모델을 미리 메모리에 올려두는 역할
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # 평가 모드 설정
            print(f"[IntruderClassifier] 모델 로드 완료: {model_path}")
        except FileNotFoundError:
            print(f"[Error] 모델 파일({model_path})을 찾을 수 없습니다!")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 학습 데이터 폴더 이름 순서 (알파벳순: intruder -> normal)
        self.classes = ['Intruder', 'Normal']

    def predict(self, frame):
        """
        외부에서 이미지(frame) 하나를 던져주면, 사람인지 아닌지 판단해서 결과 리턴
        Input: OpenCV 이미지
        Output: 결과 문자열('Intruder'/'Normal'), 확신도(%), 처리된 이미지
        """

        # [안전장치] 유효하지 않은 이미지 방지
        if frame is None or frame.size == 0:
            return "Error", 0.0, None

        # 1. 전처리 (HOG 사람 감지 + ROI 추출 + 패딩)
        processed_img = extract_intruder_roi_and_pad(frame, target_size=(128, 128))

        # 2. 형변환 (OpenCV -> PIL -> Tensor)
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        # 3. 예측
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

            result_class = self.classes[predicted.item()]
            conf_percent = confidence.item() * 100

        return result_class, conf_percent, processed_img