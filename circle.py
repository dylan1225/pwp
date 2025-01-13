import cv2
import numpy as np
import sys

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (1,1))
    gray = cv2.medianBlur(gray, 5)
    # gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.Canny(gray,115,225)
    row = frame.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp = 1, minDist = row / 8,
                               param1 = 500, param2 = 110,
                               minRadius = 1, maxRadius = 800)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(frame, center, 1, (0,0,255), 10)
            cv2.circle(frame, center, radius, (0, 250, 250), 10)
    return frame

def main(video):
    cap = cv2.VideoCapture(0)
    if video != '':
        cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("error reading frame")
            break
        frame = process_frame(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        main('')
        sys.exit(1)
    main(sys.argv[1])
