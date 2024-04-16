import cv2
import numpy as np

img1 = cv2.imread('./picture/hw2/1.jpg')
img2 = cv2.imread('./picture/hw2/2.jpg')
img3 = cv2.imread('./picture/hw2/3.jpg')
savePath = './picture/hw2/donePic/'

def changeGray():
    grayPic.append(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    grayPic.append(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    grayPic.append(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY))
    for i in range(3):
        cv2.imwrite(savePath + f'gray_{i+1}.jpg', grayPic[i])
    return

def changeFilter():
    for i in range(3):
        filterPic.append(cv2.GaussianBlur(grayPic[i], (17, 17), 0))
        cv2.imwrite(savePath + f'filter_{i + 1}.jpg', filterPic[i])
    return

def edgeDetection():
    # Laplacian(img, ddepth, ksize, scale)
    # img 來源影像
    # ddepth 影像深度
    # ksize 運算區域大小(預設1)
    # scale 縮放比例常數(預設1)
    edgePic.append(cv2.Sobel(filterPic[0], -1, 1, 1, 1, 7))
    edgePic.append(cv2.Laplacian(filterPic[1], -1, 1, 5))
    edgePic.append(cv2.Laplacian(filterPic[2], -1, 1, 5))
    for i in range(3):
        cv2.imwrite(savePath + f'edge_{i + 1}.jpg', edgePic[i])
    return 

def changeBinarization():
    ret1, binaryPic[0] = cv2.threshold(edgePic[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, binaryPic[1] = cv2.threshold(edgePic[1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret3, binaryPic[2] = cv2.threshold(edgePic[2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (ret1): cv2.imwrite(savePath + f'bin_1.jpg', binaryPic[0])
    if (ret2): cv2.imwrite(savePath + f'bin_2.jpg', binaryPic[1])
    if (ret3): cv2.imwrite(savePath + f'bin_3.jpg', binaryPic[2])

    return

def changeMorphology():
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    for i in range(3):
        morphologyPic.append(cv2.morphologyEx(binaryPic[i], cv2.MORPH_CLOSE, kernel, iterations=1))
        cv2.imwrite(savePath + f'morphology_{i + 1}.jpg', morphologyPic[i])
    
    return

def changeLines():
    lines1 = cv2.HoughLinesP(morphologyPic[0], 1, np.pi / 180, 100, minLineLength=200, maxLineGap=3)
    for line in lines1:
        x1, y1, x2, y2 = line[0]
        cv2.line(img1,(x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(savePath + f'line_1.jpg', img1)

    lines2 = cv2.HoughLinesP(morphologyPic[1], 1, np.pi / 180, 100, minLineLength=100, maxLineGap=3)
    for line in lines2:
        x1, y1, x2, y2 = line[0]
        cv2.line(img2,(x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2.imwrite(savePath + f'line_2.jpg', img2)

    lines3 = cv2.HoughLinesP(morphologyPic[2], 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5)
    for line in lines3:
        x1, y1, x2, y2 = line[0]
        cv2.line(img3,(x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(savePath + f'line_3.jpg', img3)
        
    return


grayPic = []
filterPic = []
edgePic = []
binaryPic = [0, 0, 0]
morphologyPic = []
changeGray()
changeFilter()
edgeDetection()
changeBinarization()
changeMorphology()
changeLines()