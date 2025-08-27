import cv2
import numpy as np

img_file = '25-0820.jpg'

# ====== 1) 이미지 로드 ======
img = cv2.imread(img_file)
gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

if img is not None:
    cv2.imshow('RGB', img)
    cv2.imshow('Gray', gray_img)

    # 1. Grey to Binary
    ret, bin_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY) 
    cv2.imshow('Binary', bin_img)

    # 2. black pixel count (black == 0, white == 1)
    leaf_pixels = np.sum(bin_img == 0)
    print(f"leaf pixels: {leaf_pixels} px")


    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No Image')