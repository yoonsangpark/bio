import cv2

img_file = './opencv.png'
img = cv2.imread(img_file)

if img is not None:
    cv2.imshow('IMG', img)
    cv2.imwrite('./opencv_backup.png', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No Image')