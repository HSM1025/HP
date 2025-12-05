import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image

# ★★★ 이번엔 'IntruderPreprocessing'에서 가져옵니다 ★★★
from IntruderPreprocessing import extract_intruder_roi_and_pad


# ========================================================
# 1. 커스텀 전처리 함수 정의 (학습 데이터에도 똑같이 적용!)
# ========================================================
def custom_preprocessing(pil_image):
    """
    Pytorch의 이미지(PIL)를 받아서 -> OpenCV로 바꾸고 ->
    사람 찾기(HOG) 및 전처리를 한 뒤 -> 다시 PIL로 돌려주는 함수
    """
    # 1. PIL -> OpenCV 변환
    img_np = np.array(pil_image)

    if len(img_np.shape) == 2:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. ★ 핵심: 사람 형태 찾아서 ROI 추출 (우리가 만든 함수)
    processed_cv = extract_intruder_roi_and_pad(img_cv, target_size=(128, 128))

    # 3. OpenCV -> PIL 변환
    img_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ========================================================
# 2. 설정 (Transforms)
# ========================================================
transform = transforms.Compose([
    # 침입자 전처리 적용
    transforms.Lambda(custom_preprocessing),

    # 데이터 증강 (좌우 반전은 사람 학습에 매우 효과적)
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
])

# 3. 데이터 불러오기 (폴더 이름 주의: dataset_intruder)
try:
    train_data = datasets.ImageFolder(root='./dataset_intruder/train', transform=transform)
except FileNotFoundError:
    print("[Error] 'dataset_intruder/train' 폴더가 없습니다. 폴더를 만들고 사진을 넣어주세요!")
    exit()

# 데이터가 비어있으면 에러 방지
if len(train_data) == 0:
    print("[Error] 폴더 안에 사진이 없습니다. 'intruder'와 'normal' 폴더에 사진을 넣어주세요.")
    exit()

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


# 4. CNN 모델 설계 (SimpleCNN) - 화재 모델과 구조는 동일하게 유지
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



# 5. 학습 시작
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 클래스 정보 출력 (어떤 폴더가 0번이고 1번인지 확인용)
    print(f"감지된 클래스: {train_data.classes}")
    # 예: ['intruder', 'normal']

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    print(f"침입자 감지 모델 학습을 시작합니다... (총 {epochs}회)")

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Loss: {running_loss / len(train_loader):.4f}")

    # 6. 모델 저장 (이름: my_cnn_model_2.pth)
    torch.save(model.state_dict(), 'my_cnn_model_2.pth')
    print("\n[완료] 학습이 끝났습니다!")
    print("-> 'my_cnn_model_2.pth' 파일이 생성되었습니다.")