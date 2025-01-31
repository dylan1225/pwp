import cv2
import numpy as np
import sys

#get circle from frame
def get_circle(frame):
   #get height of the frame
   rows = frame.shape[0]
   circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp = 1, minDist = rows/16, param1 = 135, param2=115, minRadius = 25, maxRadius = 1000)
   return circles


def draw_circle(frame, circles,r):
   #display the circle if there is circle
   if circles is not None:
       circles = np.uint16(np.around(circles[0, :]))
       x, y, r1 = max(circles, key=lambda circle: circle[2])
       if r != 0:
           cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)
           cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

def process_frame(frame, r):
   #blur and mask the frame
   frame = cv2.resize(frame, None, fx = 0.70, fy = 0.70)
   lower = np.array([150,150,150])
   upper = np.array([180,180,180])
   mask = cv2.inRange(frame, lower, upper)
   masked = cv2.bitwise_and(frame,frame, mask =mask)
   gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray,(5,5),0)
   blur = cv2.Canny(blur, 100, 200)
   blur = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2.5)
   cv2.imshow('blur', blur)
   circles = get_circle(blur)
   draw_circle(frame, circles, r)
   return frame, r

def main():
   #capture the video
   cap = None
   r = 0
   if len(sys.argv) < 2:
       cap = cv2.VideoCapture(0)
   else:
       cap = cv2.VideoCapture(sys.argv[1])
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           print("error reading frame")
           input('')
           break

       frame, r = process_frame(frame, r)
       cv2.imshow('frame', frame)
       if cv2.waitKey(1) == ord('q'):
           input('')
   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()


