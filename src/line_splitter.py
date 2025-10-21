import cv2
from matplotlib import pyplot as plt
import numpy as np

class Line:
    def __init__(self, char, img):
        self.char = char
        self.img = img

class LineSplitter:
    def __init__(self, path, alphabet):
        self.lines = []
        self.alphabet = alphabet

        self._process_image(path)
    
    def _process_image(self, path):
        img = self._open_image(path)
        cropped = self._crop_text_block(img)
        self._create_lines(cropped)
    
    def _create_lines(self, cropped):
        error = 10
        img_height = cropped.shape[0]
        line_height = img_height // len(self.alphabet)
        for i, char in enumerate(self.alphabet):
            linear_error = int((i/len(self.alphabet))*error)
            line_vertical_start = i * line_height - linear_error
            line_vertical_start = max(0, line_vertical_start)
            line_estimated_height = line_height + 2 * error
            line_vertical_end = min(line_vertical_start+line_estimated_height, img_height)

            line = cropped[line_vertical_start:line_vertical_end,:]
            self.lines.append(Line(char, line))

    def _crop_text_block(self, original_img):
        BORDER = 15

        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img = gray_img.copy()
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

        img = cv2.GaussianBlur(img, (33, 33), 0)
        ret, img = cv2.threshold(img, thresh=32, maxval=255, type=cv2.THRESH_BINARY)
        img = cv2.dilate(img, np.ones((2*BORDER + 1, 255), np.uint8), iterations=1)

        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(original_img, contours, -1, (0, 255, 0), 5)

        biggest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(biggest_contour)
        img2 = original_img.copy()
        cv2.rectangle(img2, (x, y), (x+w, y+h), (40, 100, 250), 2)

        cropped = gray_img[y+BORDER:y+h-BORDER, x:x+w]

        return cropped

    def _open_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_lines(self):
        return self.lines