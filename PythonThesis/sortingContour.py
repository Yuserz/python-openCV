# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2 as cv
import cv2
import os
from matplotlib import pyplot as plt

def FindContour(edge, img):
    global hull
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #Sort Contour - Get only the largest contour
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnts, boundingBoxes = sort_contours(contours, method="left-to-right")

    img2 = img.copy()

    cv2.drawContours(img2, cnts[0], -1, (0, 0, 255), thickness = 2)

    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(img2, [hull], 0, (0, 255, 0), 2)


    return img2, contours, hull



def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def draw_contour(image, c, i):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the countour number on the image
    cv.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2)

    # return the image with the contour number drawn on it
    return image



def saveImages(img):

    # for naming contoured image
    num = 1

    for i in img:
        cv.imwrite('segmentedImg/' + str(num) + '.png', i)
        num += 1


#read
img = cv.imread('image/1.jpg')
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

#blur
ks = 5
sigma = 5
blur = cv.GaussianBlur(gray, (ks, ks), sigmaX=sigma, sigmaY=sigma)

#canny
edge = cv.Canny(gray,100,150)


img3 = FindContour(edge, img)

c = 0
i = 0

contour = draw_contour(img3, c, i)




#WriteFunction
saveImages(img3)
