import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, img  = cap.read()
        if ret:
            cv2.imshow('camera', img)
            cv2.imwrite('./captured_img.png', img)
            if cv2.waitKey(1) & 0xFF == 27: #esc key
                break
        else:
            print('can not open camera')
            break
else:
    print('can not open video')
            
cap.release()
cv2.destroyAllWindows()