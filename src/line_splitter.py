import cv2
import numpy as np

class Line:
    def __init__(self, char, img):
        self.char = char
        self.img = img
    
def _process_image(path, alphabet):
    img = _open_image(path)
    line_heights = _detect_lines(img)
    classified_rows = _create_lines(img, line_heights, alphabet)

    return classified_rows

def _detect_lines(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.Canny(img, threshold1=50, threshold2=200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=60)

    heights = list(map(lambda line: (line[0][1]+line[0][3]) // 2, lines))
    sorted_heights = sorted(heights)

    separations = _get_separators(sorted_heights, threshold = 16)
    row_heights = _get_line_heights(sorted_heights, separations)

    return row_heights

def _get_separators(heights, threshold):
    indexes = [0]
    for i in range(0, len(heights)-1):
        if heights[i+1]-heights[i] > threshold:
            indexes.append(i)

    indexes.append(len(heights)-1)

    return indexes

def _get_line_heights(sorted_heights, separations):
    line_heights = []
    for i in range(0, len(separations)-1):
        idx1 = separations[i]
        idx2 = separations[i+1]
        group = sorted_heights[idx1+1:idx2]

        average = sum(group)//len(group)
        line_heights.append(average)
    
    return line_heights

def _create_lines(img, row_heights, alphabet):
    rows = {}
    for i in range(0, len(row_heights)-1):
        img_line = img[row_heights[i]:row_heights[i+1], :]
        rows[alphabet[i]] = img_line

    return rows

def _open_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    border = 50
    img = img[border:img.shape[0]-border, border:img.shape[1]-border] # crop

    return img