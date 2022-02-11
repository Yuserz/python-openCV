import cv2 as cv
import cv2
import numpy as np
import os
import PIL
from matplotlib import pyplot as plt




def saveImages(img):

    # for naming contoured image
    num = 1

    for i in img:
        cv.imwrite('segImg/' + str(num) + '.png', i)
        num += 1
    print("Saved Successfully!")

def readImg_onFolder(img, dir):

    # READ IMAGE
    #append the folder directory and filename in every loop and store to as list
    for list in dir:
        imgLoc.append('image/' + list)

    #Read all image in the list
    for j in imgLoc:
        img2 = cv.imread(str(j))
        img.append(img2)

    return img


def largestContours(edge, img):

    # Finding Contour
    contours, _ = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # PIL module used to compy original image
    orig_img = img.copy()


    # Contours -  maybe the largest perimeters pinpoint to the leaf?
    perimeter = []
    i = 0

    # Find perimeter for each contour i = id of contour
    for each_cnt in contours:
        prm = cv.arcLength(each_cnt, True)
        perimeter.append([prm, i])
        i += 1

    # Sort them
    perimeter = quick_sort(perimeter)

    unifiedContour = []
    max_index = []
    # Draw max contours
    for i in range(len(contours)):
        index = perimeter[i][1]
        max_index.append(index)
        # cv.drawContours(orig_img, contours, index, (0, 0, 255), 2)

    # Get convex hull for max contours and draw them
    cont = np.vstack(contours[i] for i in max_index)
    hull = cv.convexHull(cont)
    unifiedContour.append(hull)
    cv.drawContours(orig_img, unifiedContour, -1, (0,255,0), 2)
    boundingBoxes = [cv.boundingRect(c) for c in unifiedContour]
    print("BoundingBox:", boundingBoxes)

    return orig_img, contours, perimeter, hull, unifiedContour, boundingBoxes



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



def grCut(bd, img):
    #rectangle that contains the object
    rect = []

    #Rectangle will get the 4 index in the boundingBox of the contour
    for boundingBox in bd:
        rect = (boundingBox)


    # Our mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Values needed for algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Grabcut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    return img

# -----------------------------------------START---------------------------------------------------
#Variables

imgLoc = []
imgList = []
origImg = []
grayImg = []
blurImg = []
cannyEdge = []
contourImg = []
gCut = []
convexHull = []
boundingBoxes= []

#imagePath
path = "image"

#array of image list
dir_list = os.listdir(path)

# READ IMAGE
imgList = readImg_onFolder(imgList, dir_list)



#make a list of original image
for a in imgList:
    origImg.append(a)



#convert image to gray
for i in imgList:
    # convert to grayscale
    gray = cv.cvtColor(i, cv.COLOR_RGB2GRAY)
    grayImg.append(gray)

#Blur image
for j in grayImg:
    # {GAUSSIANBLUR VALUE} kernel size is none negative & odd numbers only
    #SMOOTHING(Applying GaussianBlur)
    ks = 5
    sigma = 5
    blur = cv.GaussianBlur(j, (ks, ks),sigmaX=sigma, sigmaY=sigma)
    blurImg.append(blur)


#Process Canny
for k in blurImg:
    # CANNY(Finding Edge)
    canny = cv.Canny(k, 5, 70, L2gradient=True)
    cannyEdge.append(canny)


#Find Contour(Find & Draw)
for c, o in zip(cannyEdge, origImg):
    # FINDING CONTOUR
    # Largest Contour - Not the best segmentation
    orig_img, contours, perimeter, hull, unifiedContour, boundingBoxes = largestContours(c, o)
    contourImg.append(orig_img)
    convexHull.append(hull)
    # Box.append(boundingBoxes)

#GrabCut the contoured Nail
for h, g in zip(boundingBoxes, origImg):
    #Cutting the contoured nail
    grbcut = grCut(h, g)
    gCut.append(grbcut)


#WriteFunction
saveImages(gCut)
# saveImages(origImg)
# saveImages(cannyEdge)
# saveImages(contourImg)
# saveImages(cannyEdge)

