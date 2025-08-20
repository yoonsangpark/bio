import cv2

video_file = 'captupred_video.avi'
cap = cv2.VideoCapture(0)
fps = 40
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(video_file, fourcc, fps, size)      

if cap.isOpened():
    while True:
        ret, img  = cap.read()
        if ret:
            cv2.imshow('camera', img)
            cv2.imwrite('./captured_img.png', img)
            out.write(img)
            if cv2.waitKey(1) & 0xFF == 27: #esc key
                break
        else:
            print('no more frame')
            break
else:
    print('can not open video')

out.release()        
cap.release()
cv2.destroyAllWindows()