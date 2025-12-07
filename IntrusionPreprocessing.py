import cv2
import numpy as np


def extract_intruder_roi_and_pad(image_cv, target_size=(128, 128)):
    """
    이미지에서 '사람 형태'를 찾아 ROI를 추출하고,
    비율을 유지한 채 검은 배경(Padding)을 추가하여 리사이징하는 함수
    """

    # [안전장치] 이미지가 비어있으면 검은 화면 반환
    if image_cv is None or image_cv.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 1. 사람 감지기 초기화 (HOG Descriptor 사용)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 2. 사람 찾기
    boxes, weights = hog.detectMultiScale(image_cv, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # 사람이 발견되지 않았으면 -> 그냥 원본을 리사이징해서 반환
    if len(boxes) == 0:
        return preprocess_padding(image_cv, target_size)

    # 3. 가장 크기가 큰 박스 선택 (단, 너무 작은 건 무시)
    best_box = None
    max_area = 0

    # ★★★ [수정됨] 최소 크기 기준 설정 ★★★
    # 이 값보다 작으면 강아지나 노이즈로 간주하고 무시합니다.
    # 영상 해상도에 따라 이 값을 조절해야 할 수도 있습니다.
    MIN_WIDTH = 30  # 사람이라기엔 너무 좁은 폭
    MIN_HEIGHT = 220  # 사람이라기엔 너무 낮은 키 (강아지는 보통 키가 작음)
    MIN_AREA = 3000  # 전체 면적 (w * h)

    for (x, y, w, h) in boxes:
        # [필터링] 너무 작은 물체(강아지 등)는 건너뛰기
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        area = w * h

        # [필터링] 전체 면적이 너무 작아도 건너뛰기
        if area < MIN_AREA:
            continue

        # 조건 통과한 것 중에서 가장 큰 것 찾기
        if area > max_area:
            max_area = area
            best_box = (x, y, w, h)

    # [예외 처리] HOG가 뭔가를 찾긴 했지만, 전부 너무 작아서 다 걸러진 경우
    if best_box is None:
        # ROI를 추출하지 않고 원본 전체를 넘김 (혹은 검은 화면을 넘길 수도 있음)
        return preprocess_padding(image_cv, target_size)

    x, y, w, h = best_box

    # [안전장치] ROI 좌표가 이미지 벗어나지 않게 클램핑
    h_img, w_img = image_cv.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    # 4. ROI 잘라내기
    roi = image_cv[y:y + h, x:x + w]

    if roi.size == 0:
        return preprocess_padding(image_cv, target_size)

    # 5. 패딩 추가 (비율 유지 리사이징)
    result_img = preprocess_padding(roi, target_size)

    return result_img


def preprocess_padding(img, target_size):
    """
    (FirePreprocessing과 동일)
    이미지 비율을 유지하면서 나머지 공간을 검은색으로 채우는 함수
    """
    if img is None or img.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    h, w = img.shape[:2]
    target_w, target_h = target_size

    if w == 0 or h == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

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