import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# ★★★ 침입자 전용 전처리 함수 가져오기 ★★★
from IntruderPreprocessing import extract_intruder_roi_and_pad

# ==========================================
# 1. 테스트하고 싶은 사진 파일 경로
# ==========================================
# 테스트할 사람 사진 경로를 여기에 적어주세요.
target_image_path = "C:/Users/User/Desktop/i4.png"


# 2. 모델 구조 정의 (train_cnn_2.py와 동일해야 함)
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

    # ★ 파일 이름 주의: my_cnn_model_2.pth
    try:
        model.load_state_dict(torch.load('my_cnn_model_2.pth', map_location=device))
        print("모델 로드 성공!")
    except FileNotFoundError:
        print("에러: 'my_cnn_model_2.pth' 파일이 없습니다. 먼저 학습(train_cnn_2.py)을 하세요.")
        return
    model.eval()

    # 4. 이미지 불러오기
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print("에러: 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # ========================================================
    # [핵심] 침입자 전처리 적용 (HOG로 사람 찾기 + 배경 검게)
    # ========================================================
    processed_img = extract_intruder_roi_and_pad(img_cv, target_size=(128, 128))

    # 5. 모델 입력 변환
    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 6. 예측 수행
    # ★ 중요: 폴더 이름이 intruder, normal 순서라면(알파벳순)
    # i(intruder)가 n(normal)보다 앞서므로 0번이 Intruder입니다.
    classes = ['Intruder', 'Normal']

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

        result_class = classes[predicted.item()]
        conf_percent = confidence.item() * 100

    # 7. 결과 보여주기
    print(f"\n[분석 결과]")
    print(f"판정: {result_class}")
    print(f"확신도: {conf_percent:.2f}%")

    # 결과 텍스트 색상 (침입자면 빨강, 정상이면 초록)
    color = (0, 0, 255) if result_class == 'Intruder' else (0, 255, 0)

    # 원본 이미지에 결과 쓰기
    cv2.putText(img_cv, f"{result_class} ({conf_percent:.1f}%)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 화면에 띄우기
    cv2.imshow("Original Result", img_cv)

    # AI가 보는 화면 3배 확대해서 보여주기
    debug_view = cv2.resize(processed_img, (384, 384), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("AI Input (Processed)", debug_view)

    print("아무 키나 누르면 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_single_image(target_image_path)