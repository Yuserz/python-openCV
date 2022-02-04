import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt


# def largestContours(canny, img, img_gray):

#     contours, hierarchy = cv2.findContours(
#         canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     img_contour = np.copy(img)  # Contours change original image.
#     # cv2.drawContours(img_contour, contours, -1, (0,255,0), 3) # Draw all - For visualization only

#     # Contours -  maybe the largest perimeters pinpoint to the leaf?
#     perimeter = []
#     max_perim = [0, 0]
#     i = 0

#     # Find perimeter for each contour i = id of contour
#     for each_cnt in contours:
#         prm = cv2.arcLength(each_cnt, False)
#         perimeter.append([prm, i])
#         i += 1

#     # Sort them
#     perimeter = quick_sort(perimeter)

#     unified = []
#     max_index = []
#     # Draw max contours
#     for i in range(0, 3):
#         index = perimeter[i][1]
#         max_index.append(index)
#         cv2.drawContours(img_contour, contours, index, (255, 0, 0), 3)

#     # Get convex hull for max contours and draw them
#     cont = np.vstack(contours[i] for i in max_index)
#     hull = cv2.convexHull(cont)
#     unified.append(hull)
#     cv2.drawContours(img_contour, unified, -1, (0, 0, 255), 3)

#     return img_contour, contours, perimeter, hull


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


def grCut(chull, img):
    # First create our rectangle that contains the object
    y_corners = np.amax(chull, axis=0)
    x_corners = np.amin(chull, axis=0)
    x_min = x_corners[0][1]
    x_max = x_corners[0][1]
    y_min = y_corners[0][1]
    y_max = y_corners[0][1]
    rect = (x_min, x_max, y_min, y_max)

    # Our mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Values needed for algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Grabcut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_PR_BGD) | (
        mask == cv2.GC_BGD), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]

    return img


img = cv.imread('testSample1.jpg')  # readimage
# img = cv.imread('image/bl.jpg') #readimage
# img = cv.imread('image/q.PNG') #readimage

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Smoothing
# kernel size is none negative & odd numbers only
ks_width = 7
ks_height = 11
sigma_x = 5
sigma_y = 5
dst = None

img_blur = cv.GaussianBlur(gray, (ks_width, ks_height), sigma_x, dst, sigma_y)
# img_blur = cv.blur(gray,(0,0),0)#Smoothing

# Canny(Finding Edge)
# Noise Reduction, Finding Intensity Gradient of the Image,
canny = cv.Canny(img_blur, 8, 20, L2gradient=True)


# Finding Contour
contours, hierarchy = cv.findContours(
    canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
img_contour = np.copy(img)  # Contours change original image.
# cv2.drawContours(img_contour, contours, -1, (0,255,0), 3) # Draw all - For visualization only

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
for i in range(8, 30):
    index = perimeter[i][1]
    max_index.append(index)
    cv.drawContours(img_contour, contours, index, (255, 0, 0), 3)

# Get convex hull for max contours and draw them
cont = np.vstack(contours[i] for i in max_index)
hull = cv.convexHull(cont)
unified.append(hull)
cv.drawContours(img_contour, unified, -1, (0, 0, 255), 3)

img_grcut = grCut(hull, img)
# cv2.imwrite('contours_none_image1.jpg', image_copy)
# cv2.destroyAllWindows()


imageArray = ([img, img_blur, canny, img_contour, img_grcut])
imageStacked = stackImages(imageArray, 0.5)

cv2.imshow("original", imageStacked)
cv2.waitKey(0)