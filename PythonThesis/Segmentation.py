# import the necessary packages
import bound as bound
import numpy as np
import imutils
import cv2
import os



def saveImages(img):

    # for naming contoured image
    num = 1

    for i in img:
        cv2.imwrite('segImg/' + str(num) + '.png', i)
        num += 1
    print("Saved Successfully!")

def readImg_onFolder(img, dir):

    # READ IMAGE
    #append the folder directory and filename in every loop and store to as list
    for list in dir:
        imgLoc.append('image/' + list)

    #Read all image in the list
    for j in imgLoc:
        img2 = cv2.imread(str(j))
        img.append(img2)

    return img


def sort_contours(contours):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                            key=lambda b:b[1][i]))

    # return the list of sorted contours and bounding boxes
    return (contours, boundingBoxes)

def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the countour number on the image
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 225), 0)

    # return the image with the contour number drawn on it
    return image

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
    cv2.grabCut(gCut, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_PR_BGD) | (
        mask == cv2.GC_BGD), 0, 1).astype('uint8')
    gCut = gCut*mask2[:, :, np.newaxis]

    return gCut


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

def findContour(edge, img):
    # find contours in the accumulated image, keeping only the largest
    # ones
    contours = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    orig = img.copy()

    # # loop over the (unsorted) contours and draw them
    # for (i, c) in enumerate(contours):
    #     orig = draw_contour(orig, c, i)

    # show the original, unsorted contour image
    # cv2.imshow("Unsorted", orig)

    # sort the contours according to the provided method
    (contours, boundingBoxes) = sort_contours(contours)

    # loop over the (now sorted) contours and draw them
    for (i, c) in enumerate(contours):
        draw_contour(img, c, i)

    # Contours -  maybe the largest perimeters pinpoint to the leaf?
    perimeter = []
    i = 0

    # Find perimeter for each contour i = id of contour
    for each_cnt in contours:
        prm = cv2.arcLength(each_cnt, True)
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
        # cv2.drawContours(orig, contours, index, (0, 0, 255), 2)

    # Get convex hull for max contours and draw them
    cont = np.vstack(contours[i] for i in max_index)
    hull = cv2.convexHull(cont)
    cv2.drawContours(orig, unified, -1, (255, 0, 0), 3)

    return orig, hull, boundingBoxes


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
    gray = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    grayImg.append(gray)

#Blur image
for j in grayImg:
    # {GAUSSIANBLUR VALUE} kernel size is none negative & odd numbers only
    #SMOOTHING(Applying GaussianBlur)
    ks = 5
    sigma = 5
    blur = cv2.GaussianBlur(j, (ks, ks),sigmaX=sigma, sigmaY=sigma)
    blurImg.append(blur)


#Process Canny
for k in blurImg:
    # CANNY(Finding Edge)
    canny = cv2.Canny(k, 5, 70, L2gradient=True)
    cannyEdge.append(canny)


bound = []
#Find Contour(Find & Draw)
for c, o in zip(cannyEdge, origImg):
    # FINDING CONTOUR
    # Largest Contour - Not the best segmentation
    contours, hull, boundingBoxes = findContour(c, o)
    contourImg.append(contours)
    convexHull.append(hull)
    bound.append(boundingBoxes)


#GrabCut the contoured Nail
for h, g in zip(convexHull, origImg):
    #Cutting the contoured nail
    grbcut = grCut(h, g)
    gCut.append(grbcut)


for con in convexHull:
    # print(bound)
    print(convexHull)


#WriteFunction
# saveImages(gCut)
# saveImages(origImg)
# saveImages(cannyEdge)
saveImages(contourImg)
# saveImages(cannyEdge)
