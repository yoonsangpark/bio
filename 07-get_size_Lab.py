import cv2
import numpy as np

# === 사용자 입력 ===
IMAGE_PATH = '25-0820.jpg'  # 사진 경로
PETRI_DIAMETER_MM = 10.0        # 실제 페트리디쉬 안쪽 지름(mm) → 맞게 수정하세요

# === 1. 이미지 불러오기 ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

# 해상도 너무 크면 줄이기
#scale = 1200 / max(img.shape[:2])
#if scale < 1.0:
#    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# === 2. 원 검출 (페트리디쉬) ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                           minDist=100, param1=100, param2=50,
                           minRadius=100, maxRadius=0)

if circles is None:
    raise RuntimeError("페트리디쉬 원을 찾지 못했습니다. 파라미터 조정 필요")

circles = np.uint16(np.around(circles))
x, y, r = circles[0][0]  # 가장 큰 원
print(f"검출된 원 중심=({x},{y}), 반지름={r}px")

# === 3. ROI 자르기 (원 안쪽만) ===
mask_circle = np.zeros_like(gray)
cv2.circle(mask_circle, (x, y), r-5, 255, -1)
masked_img = cv2.bitwise_and(img, img, mask=mask_circle)

# === 4. 색공간 변환 (LAB) ===
lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# CLAHE로 조명 보정
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
L_eq = clahe.apply(L)
lab_eq = cv2.merge([L_eq, A, B])

# 녹색 계열 분리 (a채널 음수 방향)
# A채널이 낮을수록 녹색, B채널은 노랑 성분 → 두 채널 함께 임계
_, mask_a = cv2.threshold(255 - A, 40, 255, cv2.THRESH_BINARY)
_, mask_b = cv2.threshold(B, 150, 255, cv2.THRESH_BINARY_INV)
mask = cv2.bitwise_and(mask_a, mask_b)

# 잡음 제거
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# === 5. 컨투어 기반 면적 계산 ===
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas_px = [cv2.contourArea(c) for c in contours]
total_area_px = sum(areas_px)
print(f"녹색 잎 픽셀 면적(컨투어 기반): {total_area_px:.1f} px²")

# === 6. 실제 면적 환산 ===
# (실제 지름 / 픽셀 지름) → mm/px
mm_per_px = PETRI_DIAMETER_MM / (2*r)
area_mm2 = total_area_px * (mm_per_px**2)
print(f"녹색 잎 실제 면적: {area_mm2:.2f} mm²")

# === 7. 시각화 저장 ===
vis = img.copy()
cv2.circle(vis, (x,y), r, (255,0,0), 2)
cv2.drawContours(vis, contours, -1, (0,0,255), 2)
cv2.imwrite("leaf_mask.png", mask)
cv2.imwrite("leaf_contours.png", vis)
print("결과 leaf_mask.png, leaf_contours.png 로 저장됨")