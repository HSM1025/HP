import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from FirePreprocessing import extract_fire_roi_and_pad

# ==========================================
# 1. 테스트하고 싶은 사진 파일 경로
# ==========================================
target_image_path = "C:/Users/PC/Desktop/5.jpg"


# 2. 모델 구조 정의 (건드리지 마세요)
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


def predict_single_image(image_path):
    # 3. 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    try:
        model.load_state_dict(torch.load('my_cnn_model.pth', map_location=device))
    except FileNotFoundError:
        print("에러: 모델 파일이 없습니다.")
        return
    model.eval()

    # 4. 이미지 불러오기 (OpenCV로 먼저 읽어야 전처리가 가능)
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print("에러: 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # ========================================================
    # [핵심 변경점] 전처리 함수 적용 (배경 날리고 불만 남기기)
    # ========================================================
    processed_img = extract_fire_roi_and_pad(img_cv, target_size=(128, 128))

    # 5. 모델 입력 변환 (전처리된 이미지를 사용!)
    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img_pil = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미 리사이징은 전처리 함수에서 했으므로 ToTensor만 함
    ])

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 6. 예측 수행
    classes = ['Fire', 'Non-Fire']
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

        result_class = classes[predicted.item()]
        conf_percent = confidence.item() * 100

    # 7. 결과 보여주기 (원본 + AI가 본 세상)
    print(f"\n[분석 결과]")
    print(f"판정: {result_class}")
    print(f"확신도: {conf_percent:.2f}%")

    # 결과 텍스트 색상
    color = (0, 0, 255) if result_class == 'Fire' else (0, 255, 0)

    # 원본 이미지에 결과 쓰기
    cv2.putText(img_cv, f"{result_class} ({conf_percent:.1f}%)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 창 띄우기 (두 개를 띄워드릴게요)
    cv2.imshow("Original Result", img_cv)  # 1. 결과가 적힌 원본
    cv2.imshow("AI Input (Processed)", processed_img)  # 2. 작아서 잘 안 보임

    print("아무 키나 누르면 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_single_image(target_image_path)