import cv2
import numpy as np

inputPath = './picture/hw2/1.jpg'
def change_Gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def change_Filter(img, mode):
    if mode == 0:
        return cv2.blur(img, (15, 15))
    if mode == 1:
        return cv2.GaussianBlur(img, (15, 15))
    if mode == 2:
        return cv2.medianBlur(img, 15)
    
def change_Edge(img, mode):
    if mode == 0:
        x = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        y = cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        return cv2.addWeighted(x, 0.5, y, 0.5, 0)
    if mode == 1:
        x = cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_64F, 1, 0))
        y = cv2.convertScaleAbs(cv2.Scharr(img, cv2.CV_64F, 0, 1))
        return cv2.addWeighted(x, 0.5, y, 0.5, 0)
    if mode == 2:
        return cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))
    if mode == 3:
        return cv2.Canny(img, 1, 10)
    
def change_Bin(img):
    ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def change_Morphology(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

def change_Lines(img, origin):
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=3)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(origin, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return origin

def main():
    origin_img = cv2.imread(inputPath)
    gray_img = change_Gray(origin_img)
    filter_img = change_Filter(gray_img, 1)
    edge_img = change_Edge(filter_img, 1)
    bin_img = change_Bin(edge_img)
    morphology_img = change_Morphology(bin_img)
    line_img = change_Lines(morphology_img, origin_img)

    return 0

main()