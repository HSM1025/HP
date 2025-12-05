import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from FirePreprocessing import extract_fire_roi_and_pad  # 전처리 함수


# --- 모델 구조 (학습 때와 동일) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FireClassifier:
    def __init__(self, model_path='my_cnn_model.pth'):
        """
        초기화: 모델을 미리 메모리에 올려두는 역할
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()  # 평가 모드 설정
            print(f"[FireClassifier] 모델 로드 완료: {model_path}")
        except FileNotFoundError:
            print(f"[Error] 모델 파일({model_path})을 찾을 수 없습니다!")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.classes = ['Fire', 'Non-Fire']

    def predict(self, frame):
        """
        외부에서 이미지(frame) 하나를 던져주면, 불인지 아닌지 판단해서 결과 리턴
        Input: OpenCV 이미지 (cv2.imread 등)
        Output: 결과 문자열('Fire'/'Non-Fire'), 확신도(%), 처리된 이미지
        """

        # [추가된 안전장치] 이미지가 없거나 비어있으면 에러 방지
        if frame is None or frame.size == 0:
            print("[Error] 유효하지 않은 이미지입니다.")
            return "Error", 0.0, None

        # 1. 전처리 (배경 날리기 + ROI + 패딩)
        # target_size=(128, 128)은 CNN 모델의 fc1 입력 크기와 맞아야 함 (중요!)
        processed_img = extract_fire_roi_and_pad(frame, target_size=(128, 128))

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