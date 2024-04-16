import cv2

# 讀取原圖素材 路徑須按照自己的修改
img1 = cv2.imread('./MultimediaDesign/picture/people.jpg')
img2 = cv2.imread('./MultimediaDesign/picture/pic1.jpg')
img3 = cv2.imread('./MultimediaDesign/picture/pic2.jpg')
img4 = cv2.imread('./MultimediaDesign/picture/pic3.jpg')
# 圖片要寫入的路徑
picturePath = './MultimediaDesign/picture/'

# 灰階  gray
def grayPicture():
    grayPic1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    grayPic2 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    grayPic3 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)    
    
    # 儲存圖片
    cv2.imwrite(picturePath + 'gray_pic1.jpg', grayPic1)
    cv2.imwrite(picturePath + 'gray_pic2.jpg', grayPic2)
    cv2.imwrite(picturePath + 'gray_pic3.jpg', grayPic3)
    return

# 濾波  filter
def filterPicture():
    # Blur(image, ksize)
    # image 來源影像
    # ksize 指定區域單位 (必須為大於1的奇數)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    filterPic1 = cv2.blur(img2Gray, (11, 11))

    # GaussianBlur(image, ksize, sigmaX, sigmaY)
    # image 來源影像
    # ksize 指定區域單位 (必須為大於1的奇數)
    # sigmaX X 方向標準差，預設 0，sigmaY Y 方向標準差，預設 0
    img3Gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    filterPic2 = cv2.GaussianBlur(img3Gray, (11, 11), 0)
    
    # medianBlur(image, ksize)
    # image 來源影像
    # ksize 模糊程度 (必須為大於1的奇數)
    img4Gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    filterPic3 = cv2.medianBlur(img4Gray, 11)
    
    # 儲存圖片
    cv2.imwrite(picturePath + 'filter_pic1.jpg', filterPic1)
    cv2.imwrite(picturePath + 'filter_pic2.jpg', filterPic2)
    cv2.imwrite(picturePath + 'filter_pic3.jpg', filterPic3)
    return

# 二值化    binarization
def BinaryPicture():
    # ret, output = cv2.threshold(image, thresh, maxval, type)
    # ret 是否成功轉換 成功 True 失敗 False
    # img 來源影像
    # thresh 閾值 通常設定 127
    # maxval 最大灰度 通常設定 255
    # type 轉換方式

    # Global Thresholding
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret1, binaryPic1 = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY)

    # Otsu's Threshold
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    ret2, binaryPic2 = cv2.threshold(img3_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Gaussian filter + Otsu's Threshold
    img4_gray = cv2.GaussianBlur(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ret3, binaryPic3 = cv2.threshold(img4_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 儲存圖片
    cv2.imwrite(picturePath + 'binarization_pic1.jpg', binaryPic1)
    cv2.imwrite(picturePath + 'binarization_pic2.jpg', binaryPic2)
    cv2.imwrite(picturePath + 'binarization_pic3.jpg', binaryPic3)
    return
# OTSU
# url : https://www.wongwonggoods.com/all-posts/python/python-image-process/python_opencv/opencv-threshold-all-otsu/

# 形態學    morphology
def MorphologyPicture():
    # getStructuringElement(shape, ksize)
    # 返回指定大小形狀的結構元素
    # shape 的內容：cv2.MORPH_RECT ( 矩形 )、cv2.MORPH_CROSS ( 十字交叉 )、cv2.MORPH_ELLIPSE ( 橢圓形 )
    # ksize 的格式：(x, y)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    # morphologyEx(image, type, kernel, iterations)
    # iterations 表示實施次數
    img2Gray = cv2.GaussianBlur(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ret1, binaryImg2 = cv2.threshold(img2Gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img2Opening = cv2.morphologyEx(binaryImg2, cv2.MORPH_OPEN, kernel, iterations=1)
    
    img3Gray = cv2.GaussianBlur(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ret2, binaryImg3 = cv2.threshold(img3Gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img3Closing = cv2.morphologyEx(binaryImg3, cv2.MORPH_CLOSE, kernel, iterations=1)

    img4Gray = cv2.GaussianBlur(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ret3, binaryImg4 = cv2.threshold(img4Gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img4Opening = cv2.morphologyEx(binaryImg4, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 儲存圖片
    cv2.imwrite(picturePath + 'morphology_pic1.jpg', img2Opening)
    cv2.imwrite(picturePath + 'morphology_pic2.jpg', img3Closing)
    cv2.imwrite(picturePath + 'morphology_pic3.jpg', img4Opening)
    return

#繪製形狀和文字
def DrawPicture():
    # rectangle(image, pt1, pt2, color, thickness)
    # img 來源影像
    # pt1 左上座標 pt2 右下座標
    # color 線條顏色 使用 BGR
    # thickness 線條粗細 預設 1 設定 -1 表示填滿
    cv2.rectangle(img1, (162, 295), (375, 559), (0, 255, 0), 3)
    # putText(image, text, org, fontFace, fontScale, color, thickness, lineType)
    # img 來源影像
    # text 文字內容
    # org 文字座標 ( 垂直方向是文字底部到影像頂端的距離 )
    # fontFace 文字字型
    # fontScale 文字尺寸
    # color 線條顏色 使用 BGR
    # thickness 文字外框線條粗細 預設 1
    # lineType 外框線條樣式 預設 cv2.LINE_8 設定 cv2.LINE_AA 可以反鋸齒
    cv2.putText(img1, "A Pretty Woman", (160, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 3, cv2.LINE_AA)
    # 儲存圖片
    cv2.imwrite(picturePath + 'label_people.jpg', img1)
    return

grayPicture()       # 生成灰階圖片
filterPicture()     # 生成濾波圖片
BinaryPicture()     # 生成二值化圖片
MorphologyPicture() # 生成形態學圖片
DrawPicture()       # 生成繪製形狀和文字在圖片