import numpy as np
import cv2

video = cv2.VideoCapture(0)

number = 1

while True:
    # Capture current frame
    ret, frame = video.read()

    if ret == True:
        cv2.imshow('Camera', frame)
        
        key = cv2.waitKey(1)
        # If user hit 's'
        if (key & 0xFF) == ord('s'):
            if (number < 10):
                fname = 'PImg0' + str(number) + '.jpg'
            else:
                fname = 'PImg' + str(number) + '.jpg'
            cv2.imwrite(fname, frame)
            number = number + 1
        # If user hit 'q'
        if (key & 0xFF) == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()

