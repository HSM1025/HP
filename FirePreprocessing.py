import cv2
import numpy as np


def extract_fire_roi_and_pad(image_cv, target_size=(256, 256)):
    """
    이미지에서 불 색상 영역을 찾아 ROI를 추출하고,
    비율을 유지한 채 검은 배경(Padding)을 추가하여 리사이징하는 함수
    """

    # [안전장치 1] 들어온 이미지가 없거나 깨졌으면 검은 화면 반환
    if image_cv is None or image_cv.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 1. HSV 변환 및 화재 색상 마스킹 (배경 날리기)
    try:
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    except cv2.error:
        # 변환 중 에러나면 검은 화면 반환
        return preprocess_padding(image_cv, target_size)

    # 화재 색상 범위 정의
    lower_fire1 = np.array([0, 150, 180])
    upper_fire1 = np.array([30, 255, 255])
    lower_fire2 = np.array([160, 150, 180])
    upper_fire2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 2. 동적 ROI 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선이 없으면(불 색깔이 하나도 없으면)
    if not contours:
        return preprocess_padding(image_cv, target_size)

    # 가장 큰 영역 찾기
    try:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
    except ValueError:
        return preprocess_padding(image_cv, target_size)

    # 너무 작은 노이즈 무시
    if w < 15 or h < 15:
        return preprocess_padding(image_cv, target_size)

    # ROI 잘라내기 (이미지 범위를 벗어나지 않게 클램핑)
    # 가끔 계산 오차로 범위를 벗어나면 에러가 나므로 안전장치 추가
    h_img, w_img = image_cv.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    roi = image_cv[y:y + h, x:x + w]

    # [안전장치 2] 잘라낸 ROI가 비어있으면 원본 반환
    if roi.size == 0:
        return preprocess_padding(image_cv, target_size)

    # 3. 패딩 추가
    result_img = preprocess_padding(roi, target_size)

    return result_img


def preprocess_padding(img, target_size):
    """
    이미지 비율을 유지하면서 나머지 공간을 검은색으로 채우는 함수
    """
    # [안전장치 3] 입력 이미지가 없으면 검은 도화지 반환
    if img is None or img.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 0으로 나누기 방지
    if w == 0 or h == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 크기가 너무 작아져서 0이 되면 1로 보정
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    try:
        resized_img = cv2.resize(img, (new_w, new_h))
    except cv2.error:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return canvas