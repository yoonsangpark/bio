import cv2
import numpy as np

# ====== 1) 이미지 로드 ======
img = cv2.imread('25-0820.jpg')  # <- 사진 경로
if img is None:
    raise FileNotFoundError("FileNotFoundError")

# ====== 2) 녹색 잎 마스크 만들기 (HSV) ======
# Hue(색상) Saturation(채도) Value(명도)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Hue(25~95): 노란빛(25 근처) → 청록(95 근처)까지 포함
#Saturation(20 이상): 채도가 너무 낮으면 배경 흰색까지 섞일 수 있어서 20으로 제한
#Value(20 이상): 어두운 그림자 영역 배제
lower_green = np.array([25, 20, 20])   # 범위 넓힘
upper_green = np.array([95, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# ====== 3) 마스크 정제 (노이즈 제거 & 구멍 메우기) ======
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 가장 큰 녹색 영역만 사용하고 싶다면(바닥·초록 배경 제거용):
#num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
#if num_labels > 1:
#    # 0은 배경, 1..N은 라벨, 면적 최대 라벨 선택
#    areas = stats[1:, cv2.CC_STAT_AREA]
#    max_label = 1 + np.argmax(areas)
#    mask_biggest = np.zeros_like(mask)
#    mask_biggest[labels == max_label] = 255
#    mask = mask_biggest

# ====== 4) 픽셀 면적 계산 ======
leaf_pixels = int(np.count_nonzero(mask))
print(f"잎 픽셀 면적: {leaf_pixels} px")


# ====== 5) 결과 시각화 저장 ======
vis = img.copy()
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(vis, contours, -1, (0,0,255), 2)
cv2.imwrite('leaf_mask.png', mask)
cv2.imwrite('leaf_contours.png', vis)
