import cv2 as cv
import cv2
import numpy as np
import math


def saveImages(img):
    # for naming contoured image
    num = 1

    for i in img:
        cv.imwrite('segImg/' + str(num) + '.jpg', i, (cv.IMWRITE_JPEG_QUALITY, 100))
        num += 1

    print("Saved Successfully!")

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
    global approx, unified, rect
    global hierarchy, contours
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contoured_img = np.copy(img)  # Contours change original image.


    # Get all coordinates
    coordinates = []
    i = 0

    # Find coordinates for each contour i = id of contour
    for each_cnt in contours:
        prm = cv.arcLength(each_cnt, True)
        coordinates.append([prm, i])
        i += 1

    global approx
    #Approximate Contour
    for cnt in contours:
        epsilon = 0.01 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
    print("approx",approx)


    unified = []
    max_index = []
    # Draw all contours
    for i in range(len(contours)):
        index = coordinates[i][1]
        max_index.append(index)
        # cv.drawContours(contoured_img, contours, index, (0, 0, 255), 2)

    # apply convexhull function to get the hull value then draw
    conContour = np.vstack(contours[i] for i in max_index)
    hull = cv.convexHull(conContour)
    unified.append(hull)
    cv.drawContours(contoured_img, unified,-1, (0,255,0), 2)

    # print("hull",coordinates)

    #Boundingbox will be the final perimmeter of the image
    boundingBoxes = [cv.boundingRect(uni) for uni in unified]
    print("BoundingBox:", boundingBoxes)


    for boundingBox in boundingBoxes:
        rect = boundingBox

    # Getting ROI(region of interest)
    # up1,down3,left0,right2
    # x,y,w,h
    global conLength
    global conWidth

    conLength = rect[0] + rect[2]
    conWidth= rect[1] + rect[3]

    global roi
    roi = img[rect[1]: conWidth, rect[0]:conLength]
    print(roi)


    return contoured_img, approx, coordinates, hull, unified, boundingBoxes,contours


def grCut(image, bd):
    #rectangle that contains the object

    #Rectangle will get the 4 index in the boundingBox of the contour
    global rect

    for boundingBox in bd:
        rect = (boundingBox)


    #Create 2 mask

    #Rectangular mask
    global  rec
    rec= np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(rec,rect, 255, -1)
    # cv2.imshow("Rectangular Mask", rec)

    #  circle mask
    circle = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(circle, (cx, cy), int(radius) , 255, -1) # subtracted 10 to original radius to eliminate excess pixels
    # cv2.imshow("Circle mask", circle)

    # combined using bitwise_and operator
    mask = cv2.bitwise_and(rec, circle)
    # cv2.imshow("mask", mask)
    # cropped out
    masked = cv2.bitwise_and(image, image, mask=mask)

    # # Getting ROI(region of interest)
    # # up1,down3,left0,right2
    # global roi
    # roi = image[rect[1]: rect[1] + rect[3], rect[0]:rect[2]]

    return masked

    # # Values needed for algorithm
    # bgdModel = np.zeros((0), np.float64)
    # fgdModel = np.zeros((1, 65), np.float64)
    #
    # # Grabcut
    # cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    #
    # mask2 = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
    # img_grcut = image*mask2[:, :, np.newaxis]
    #
    # return  img_grcut

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
    # cv.circle(contoured_img, (cx, cy), 5, (255, 0, 0), -1)

    # solving Area
    conArea = M["m00"]

    # Solving the radius using area
    pi = 3.14159
    area = conArea

    radius = math.sqrt(area / pi)


    # solve for perimeters p = l * w
    perimeter = 2*(conLength + conWidth)

    # Solving "Circularity or Roundness"
    Roundness = (4 * area * pi)/(perimeter * 2)

    # Solving Compactness
    compactness = (conLength*2)/conArea

    print("Perimeter",perimeter)
    print("Roundness",Roundness)
    print("Radius",radius)
    print("Area", conArea)
    print("Compactness", compactness)

    return M, cx, cy, area, radius, perimeter,Roundness, compactness


# ------------------------------------------START------------------------------------------------------
# READ IMAGE

img = cv.imread('image/5.jpg')

rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# {GAUSSIANBLUR VALUE} kernel size is none negative & odd numbers only
ks = 5
sigma= 5
#SMOOTHING(Applying GaussianBlur)
img_blur = cv.GaussianBlur(gray, (ks, ks), sigma)

# sobel filter
x = cv2.Sobel(img_blur,cv2.CV_64F, 1,0, ksize=3, scale=1)
y = cv2.Sobel(img_blur,cv2.CV_64F, 1,1, ksize=3, scale=1)
absx= cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
cv2.imshow('edge', edge)

# CANNY(Finding Edge)
canny = cv.Canny(edge, 50,150 , L2gradient= True)


# FINDING CONTOUR
# Largest Contour - Not the best segmentation
contoured_img, approx, coordinates, hull, unified, boundingBoxes,contours = largestContours(canny, img)

#Contour Analysis
M, cx, cy, area, radius, conLength, perimeter, Roundness = contourAnalysis(unified)

#masking
masked = grCut(img, boundingBoxes)

imageArray = ([img, img_blur, canny, contoured_img,masked])
# imageArray = ([img, img_blur, canny, contoured_img, masked, roi])
imageStacked = stackImages(imageArray, 0.5)


cv2.imshow("original", imageStacked)

cv2.imshow("Roi", roi)

cv2.waitKey(0)


