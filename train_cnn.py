import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image

# 전처리 도구 가져오기
from FirePreprocessing import extract_fire_roi_and_pad


# ========================================================
# 1. 커스텀 전처리 함수 정의 (학습 데이터에도 똑같이 적용!)
# ========================================================
def custom_preprocessing(pil_image):
    """
    Pytorch의 이미지(PIL)를 받아서 -> OpenCV로 바꾸고 ->
    배경 날리기(전처리)를 한 뒤 -> 다시 PIL로 돌려주는 함수
    """
    # 1. PIL 이미지를 numpy 배열(OpenCV 호환)로 변환
    img_np = np.array(pil_image)

    # 2. 색상 변환 (PIL은 RGB, OpenCV는 BGR을 씀)
    # 이미지가 흑백(채널이 2개)인 경우를 대비해 예외처리
    if len(img_np.shape) == 2:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 3. ★ 핵심: 우리가 만든 전처리 함수 실행 (배경 제거 + 리사이징)
    # 여기서 이미 128x128 크기로 맞춰져서 나옵니다.
    processed_cv = extract_fire_roi_and_pad(img_cv, target_size=(128, 128))

    # 4. 다시 PIL 이미지로 변환 (Pytorch가 이해할 수 있게)
    img_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ========================================================
# 2. 설정 (Transforms)
# ========================================================
transform = transforms.Compose([
    # (중요) 단순 Resize 대신 우리가 만든 전처리 함수를 끼워 넣음!
    transforms.Lambda(custom_preprocessing),

    # 데이터가 적을 때 성능 올리는 꿀팁 (좌우 반전, 살짝 회전 추가)
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ToTensor(),
])

# 3. 데이터 불러오기
# 데이터 폴더가 없으면 에러 나니까 try-except로 감싸서 안내해줌
try:
    train_data = datasets.ImageFolder(root='./dataset/train', transform=transform)
except FileNotFoundError:
    print("[Error] 'dataset/train' 폴더가 없습니다. 사진 데이터부터 준비해주세요!")
    exit()

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


# 4. CNN 모델 설계 (SimpleCNN) - 다른 파일들과 똑같아야 함!
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
if __name__ == '__main__':  # 윈도우에서 멀티프로세싱 에러 방지용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20  # 데이터가 적으니까 10번 말고 20번 공부시킵시다!
    print(f"학습을 시작합니다... (총 {epochs}회)")

    model.train()  # 학습 모드 전환
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

    # 6. 모델 저장
    torch.save(model.state_dict(), 'my_cnn_model.pth')
    print("\n[완료] 학습이 끝났습니다!")
    print("-> 'my_cnn_model.pth' 파일이 생성되었습니다.")