import cv2 as cv
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

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

def largestContours(canny, img):
    # Finding Contour
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contoured_img = np.copy(img)  # Contours change original image.


    # Get all perimeter
    perimeter = []
    i = 0

    # Find perimeter for each contour i = id of contour
    for each_cnt in contours:
        prm = cv.arcLength(each_cnt, True)
        perimeter.append([prm, i])
        i += 1

    # Sort perimeter and return 1 array with 4 points only
    perimeter = quick_sort(perimeter)

    unified = []
    max_index = []
    # Draw all contours
    for i in range(len(contours)):
        index = perimeter[i][1]
        max_index.append(index)
        # cv.drawContours(contoured_img, contours, index, (0, 0, 255), 2)

    # Get convexhull for max contours and draw them
    conContour = np.vstack(contours[i] for i in max_index)
    hull = cv.convexHull(conContour)
    unified.append(hull)
    cv.drawContours(contoured_img, unified,-1, (0,255,0), 2)

    #Boundingbox will be the final perimmeter of the image
    boundingBoxes = [cv.boundingRect(c) for c in unified]
    print("BoundingBox:", boundingBoxes)

    return contoured_img, contours, perimeter, hull, unified, boundingBoxes

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

def grCut(image, bd):
    #rectangle that contains the object

    #Rectangle will get the 4 index in the boundingBox of the contour
    global rect
    for boundingBox in bd:
        rect = (boundingBox)

    #split the Perimeter of boundingBox
    coordinates = np.array_split(rect, 2)
    # print(coordinates)

    # len(coordinates)

    #store to point variable
    pt1 = []
    pt2 = []
    n = 0
    for c in coordinates:
        if n == 0:
            pt1 = c
            n +=1
        else:
            pt2 = c

    # print(pt1,pt2)


    #Create 2 mask
    #Rectangular mask
    rec = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(rec,(pt1), (pt2), 255, -1)
    # cv2.imshow("Rectangular Mask", rec)

    # circle mask
    circle = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(circle, (cx, cy), int(radius) - 10, 255, -1) # subtracted 10 to original radius to eliminate excess pixels
    # cv2.imshow("Circle mask", circle)

    #combined using bitwise_and operator
    mask = cv2.bitwise_and(rec, circle)
    # cv2.imshow("mask", mask)

    # apply our mask -- notice how only the person in the image is
    # cropped out
    masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Mask Applied to Image", masked)


    # # # Our mask
    # mask = np.zeros(gCut.shape[:2], np.uint8)

    # Values needed for algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Grabcut
    cv2.grabCut(image, masked, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
    img = image*mask2[:, :, np.newaxis]

    return img

def contourAnalysis(unified):
    # Contour Analysis
    global cx, cy
    for contour in unified:
        # Get the image moment for contour
        M = cv.moments(contour)

        # Calculate the centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    # Draw a circle to indicate the contour
    cv.circle(contoured_img, (cx, cy), 5, (255, 0, 0), -1)

    # solving Area
    areaCon = M["m00"]

    print("Area", areaCon)

    # Solving the radius using area
    pi = 3.14159
    area = areaCon

    radius = math.sqrt(area / pi)

    print(radius)

    return M, cx, cy, area, radius

# ------------------------------------------START------------------------------------------------------
# READ IMAGE

img = cv.imread('image/4.jpg')

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# {GAUSSIANBLUR VALUE} kernel size is none negative & odd numbers only
ks = 5
sigma= 50
#SMOOTHING(Applying GaussianBlur)
img_blur = cv.GaussianBlur(gray, (ks, ks), sigma)

# CANNY(Finding Edge)
canny = cv.Canny(img_blur, 30,70 , L2gradient=True)

# FINDING CONTOUR
# Largest Contour - Not the best segmentation
contoured_img, contours, perimeters, hull, unified, boundingBoxes = largestContours(canny, img)

#Contour Analysis
M, cx, cy, area, radius = contourAnalysis(unified)


#Show image
# plt.figure(figsize=[10,10])
# plt.imshow(img_contour[:,:,::-1]),plt.axis("off")


#Cutting the contoured nail
img_grcut = grCut(img, boundingBoxes)


imageArray = ([img, img_blur, canny, contoured_img, img_grcut])
imageStacked = stackImages(imageArray, 0.5)

cv2.imshow("original", imageStacked)
cv2.waitKey(0)
