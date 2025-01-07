import cv2
import numpy as np

def compute_slope(line):
    x1, y1, x2, y2 = line

    if x2 - x1 == 0:
        return 0

    return (y2 - y1) / (x2 - x1)


def compute_length(line):
    x1, y1, x2, y2 = line

    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def update(line, x1, y1, x2, y2, slope):
    x1 = min(x1, line[0])
    x2 = max(x2, line[2])

    if slope > 0:
        y2 = max(y2, line[3])
        y1 = min(y1, line[1])
    else:
        y2 = min(y2, line[3])
        y1 = max(y1, line[1])

    return x1, y1, x2, y2

def draw_line(line, frame):
	cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 250, 250), 10)

def compute_line(lines):
    used = [False] * len(lines)
    ans = []

    for i in range(len(lines)):
        if used[i] == False:
            used[i] = True

            x1, y1, x2, y2 = lines[i][0]
            base_slope = compute_slope(lines[i][0])

            if base_slope < 0.1 and base_slope > -0.1:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])

                    if (
                        used[j] == False
                        and temp < 0.1
                        and temp > -0.1
                        and lines[j][0][1] < y1 + 40
                        and lines[j][0][1] > y1 - 40
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True

                ans.append([x1, int((y1 + y2) / 2), x2, int((y1 + y2) / 2)])
            else:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j][0])

                    if (
                        used[j] == False
                        and temp < base_slope + 0.19
                        and temp > base_slope - 0.19
                        # and ((x1 < lines[j][0][0] + 40 and x1 > lines[j][0][0] - 40) or (x2 < lines[j][0][2] + 40 and x2 > lines[j][0][2] - 40)) 
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j][0], x1, y1, x2, y2, temp
                        )
                        used[j] = True
                ans.append([x1, y1, x2, y2])
    return ans

def compute_line1(lines, final):
    used = [False] * len(lines)
    ans = []

    for i in range(len(lines)):
        if used[i] == False:
            used[i] = True

            x1, y1, x2, y2 = lines[i]
            base_slope = compute_slope(lines[i])

            if base_slope < 0.1 and base_slope > -0.1:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j])

                    if (
                        used[j] == False
                        and temp < 0.1
                        and temp > -0.1
                        and lines[j][1] < y1 + 40
                        and lines[j][1] > y1 - 40
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j], x1, y1, x2, y2, temp
                        )
                        used[j] = True

                ans.append([x1, int((y1 + y2) / 2), x2, int((y1 + y2) / 2)])
            else:
                for j in range(i, len(lines)):
                    temp = compute_slope(lines[j])

                    if (
                        used[j] == False
                        and temp < base_slope + 0.19
                        and temp > base_slope - 0.19
                        # and ((x1 < lines[j][0][0] + 40 and x1 > lines[j][0][0] - 40) or (x2 < lines[j][0][2] + 40 and x2 > lines[j][0][2] - 40)) 
                    ):
                        x1, y1, x2, y2 = update(
                            lines[j], x1, y1, x2, y2, temp
                        )
                        used[j] = True
                ans.append([x1, y1, x2, y2])
    if final:
    	lans = []
    	for line in ans:
    		if len(lans) < 2:
    			lans.append(line)
    		else:
    			lans.sort(key=lambda temp: compute_length(temp))
    			if compute_length(line) > compute_length(lans[0]):
    				lans[0] = line

    	return lans
    else:
    	return ans
def compute_center(lines, frame):
	neg = [0, 0, 0, 0]
	pos = [0, 0, 0, 0]

	for line in lines:
		if compute_slope(line) > -0.2 and compute_length(line) > compute_length(pos):
			pos = line.copy()
		elif compute_slope(line) < 0.2 and compute_length(line) > compute_length(neg):
			neg = line.copy()

	positive_slope = compute_slope(pos)
	negative_slope = compute_slope(neg)

	if (
		abs(negative_slope) > abs(positive_slope) - 1.5
		and abs(negative_slope) < abs(positive_slope) + 1.5):
		if neg[3] > pos[1]:
			if negative_slope != 0:
				neg[2] = int(neg[2] - (neg[3] - pos[1]) / negative_slope)
			neg[3] = pos[1]
		else:
			if positive_slope != 0:
				pos[0] = int(pos[0] + (neg[3] - pos[1]) / positive_slope)
			pos[1] = neg[3]

		if neg[1] < pos[3]:
			if negative_slope != 0:
				neg[0] = int(neg[0] - (neg[1] - pos[3]) / negative_slope)
			neg[1] = pos[3]
		else:
			if positive_slope != 0:
				pos[2] = int(pos[2] + (neg[1] - pos[3]) / positive_slope)
			pos[3] = neg[1]
		temp = int(
			(int((pos[2] + neg[0]) / 2) + int((pos[0] + neg[2]) / 2)) / 2
		)
		cv2.line(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 250, 250), 10)
		cv2.line(frame, (neg[0], neg[1]), (neg[2], neg[3]), (0, 250, 250), 10)
		return [temp + 1, pos[1], temp, pos[3]]
	else:
		return [0, 0, 0, 0]

def per_tran(lines, frame, center, lines1):
	src = [
		[lines[0][0], lines[0][1]],
		[lines[1][0], lines[1][1]], 
		[lines[0][2], lines[0][3]], 
		[lines[1][2], lines[1][3]]
		]
	if not (src[0][0] > src[1][0] - 140 and src[0][0] < src[1][0] + 140):
		center = compute_center(lines1, frame)
		if compute_slope(center) != 0:
			temp = np.zeros_like(frame)
			draw_line(center, temp)
			frame = cv2.addWeighted(frame, 1, temp, 1, 0)
		return frame
	# print(src, end = '\n\n\n')
	src = np.float32(src)
	dst = np.float32([[0,0], [frame.shape[1], 0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]]]) 
	res = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), dsize = (frame.shape[1], frame.shape[0]), flags = cv2.INTER_LINEAR)
	cv2.imshow("warp", res)
	res = np.zeros_like(frame)
	cv2.line(res, (res.shape[1]//2, 0), (res.shape[1]//2, res.shape[0]), (0, 250, 250), 40)
	res = cv2.warpPerspective(res, cv2.getPerspectiveTransform(dst, src), dsize = (frame.shape[1], frame.shape[0]), flags = cv2.INTER_LINEAR)
	ans = cv2.addWeighted(frame, 1, res, 1, 0)
	return ans


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=20
    )

    n_lines = []
    centerline = []
    # if lines is not None:
    #     for line in lines:
    #         print(compute_slope(line[0]))
    if lines is not None:
        n_lines1 = compute_line(lines)
        n_lines1 = compute_line1(n_lines1, False)
        n_lines = compute_line1(n_lines1, True)
        if len(n_lines) == 2:
        	left = abs(compute_slope(n_lines[0]))
        	right = abs(compute_slope(n_lines[1]))
        	if right < left + 1.5 and right > left - 1.5 and compute_length(n_lines[0]) < compute_length(n_lines[1]) + 100 and compute_length(n_lines[0]) > compute_length(n_lines[1]) - 100:
        		draw_line(n_lines[0], frame)
        		draw_line(n_lines[1], frame)
        		frame = per_tran(n_lines, frame, centerline, n_lines1)

    return frame

def main():
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			print("error reading frame")
			break
		frame = process_frame(frame)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) == ord('q'):
			break
		# input('')
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()