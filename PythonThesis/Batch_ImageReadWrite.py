import cv2 as cv
import glob
import os
from matplotlib import pyplot as plt

def saveImages(img):

    # for naming contoured image
    num = 1

    for i in img:
        cv.imwrite('Contoured/' + str(num) + '.png', i)
        num += 1

def readImg_onFolder(img, dir):

    # READ IMAGE
    #append the folder directory and filename in every loop and store to as list
    for list in dir:
        imgLoc.append('image/' + list)

    #Read all image in the list
    for j in imgLoc:
        img2 = cv.imread(str(j))
        cv.resize(img2, (50, 50))  # Resize images
        img.append(img2)

    return img

#imagePath
path = "image"

#array of image list
dir_list = os.listdir(path)

imgLoc = []
img = []

#ReadFunction
readImg_onFolder(img, dir_list)

#WriteFunction
saveImages(img)

