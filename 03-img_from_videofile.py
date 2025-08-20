import cv2

video_file = 'big_buck.avi'
cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    while True:
        ret, img  = cap.read()
        if ret:
            cv2.imshow('video file', img)
            cv2.imwrite('./capture.png', img)
            if cv2.waitKey(1) & 0xFF == 27: #esc key
                break
        else:
            print('no more frame')
            break
else:
    print('can not open video')
        
cap.release()
cv2.destroyAllWindows()