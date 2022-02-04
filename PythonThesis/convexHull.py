from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color)
        cv.drawContours(drawing, hull_list, i, color)
    # Show in a window
    cv.imshow('Contours', drawing)
# Load source image
# Convert image to gray and blur it


src = cv.imread('image/bl.jpg') #readimage

# convert to grayscale
gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

#Smoothing
#kernel size is none negative & odd numbers only
ks_width = 7
ks_height = 13
sigma_x = 5
sigma_y = 5
dst = None

src_gray = cv.GaussianBlur(gray,(ks_width, ks_height),sigma_x,dst,sigma_y)
# img_blur = cv.blur(gray,(0,0),0)#Smoothing


# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 200
thresh = 100 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()