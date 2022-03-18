import numpy as np
import cv2

def skeletonize(image_in):
    '''Inputs and grayscale image and outputs a binary skeleton image'''
    size = np.size(image_in)
    skel = np.zeros(image_in.shape, np.uint8)

    ret, image_edit = cv2.threshold(image_in, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    while not done:
        eroded = cv2.erode(image_edit, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image_edit, temp)
        skel = cv2.bitwise_or(skel, temp)
        image_edit = eroded.copy()

        zeros = size - cv2.countNonZero(image_edit)
        if zeros == size:
            done = True

    return skel


img = cv2.imread('image/1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img = skeletonize(gray)

cv2.imshow("mask", img)