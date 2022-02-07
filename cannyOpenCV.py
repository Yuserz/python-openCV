import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt


def largestContours(canny, img):
    # Finding Contour
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    img_contour = np.copy(img)  # Contours change original image.


    # Contours -  maybe the largest perimeters pinpoint to the leaf?
    perimeter = []
    max_perim = [0, 0]
    i = 0

    # Find perimeter for each contour i = id of contour
    for each_cnt in contours:
        prm = cv.arcLength(each_cnt, True)
        perimeter.append([prm, i])
        i += 1

    # Sort them
    perimeter = quick_sort(perimeter)

    unified = []
    max_index = []
    # Draw max contours
    for i in range(len(contours)):
        index = perimeter[i][1]
        max_index.append(index)
        # cv.drawContours(img_contour, contours, index, (0, 0, 255), 2)

    # Get convex hull for max contours and draw them
    cont = np.vstack(contours[i] for i in max_index)
    hull = cv.convexHull(cont)
    unified.append(hull)
    cv.drawContours(img_contour, unified, -1, (0,255,0), 2)

    return img_contour, contours, perimeter, hull


def quick_sort(p):
    if len(p) <= 1:
        return p

    pivot = p.pop(0)
    low, high = [], []
    for entry in p:
        if entry[0] > pivot[0]:
            high.append(entry)
        else:
            low.append(entry)
    return quick_sort(high) + [pivot] + quick_sort(low)


def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth+len(
                    lables[d][c])*13+27, 30+eachImgHeight*d), (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth*c+10, eachImgHeight *
                            d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver



def grCut(chull, gCut):
    # First create our rectangle that contains the object
    y_corners = np.amax(chull, axis=0)
    x_corners = np.amin(chull, axis=0)
    x_min = x_corners[0][1]
    x_max = x_corners[0][1]
    y_min = y_corners[0][1]
    y_max = y_corners[0][1]
    rect = (x_min, x_max, y_min, y_max)

    # Our mask
    mask = np.zeros(gCut.shape[:2], np.uint8)

    # Values needed for algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Grabcut
    cv2.grabCut(gCut, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_PR_BGD) | (
    mask == cv2.GC_BGD), 0, 1).astype('uint8')
    gCut = gCut*mask2[:, :, np.newaxis]

    return gCut



# READ IMAGE
# img = cv.imread('image/test (1).jpg')
img = cv.imread('image/1.jpg')

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# {GAUSSIANBLUR VALUE} kernel size is none negative & odd numbers only
ks = 5
sigma= 50
#SMOOTHING(Applying GaussianBlur)
img_blur = cv.GaussianBlur(gray, (ks, ks), sigma)

# CANNY(Finding Edge)
canny = cv.Canny(img_blur, 10 ,70 , L2gradient=True)

# FINDING CONTOUR
# Largest Contour - Not the best segmentation
img_contour, contours, perimeters, hull = largestContours(canny, img)

#Cutting the contoured nail
gCut = img_contour
img_grcut = grCut(hull, gCut)


imageArray = ([img, img_blur, canny, img_contour, img_grcut])
imageStacked = stackImages(imageArray, 0.5)

cv2.imshow("original", imageStacked)
cv2.waitKey(0)
