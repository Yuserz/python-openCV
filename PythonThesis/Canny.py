import cv2 as cv
import glob
import os
from matplotlib import pyplot as plt

def saveImages(img):

    # for naming contoured image
    num = 1

    for i in img:
        cv.imwrite('segmentedImg/' + str(num) + '.png', i)
        num += 1

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

#imagePath
path = "image"

#array of image list
dir_list = os.listdir(path)

imgLoc = []
img = []
img3 = []

#ReadFunction
img = readImg_onFolder(img, dir_list)


for i in img:
    img = cv.Canny(i,100,150)
    img3.append(img)


#WriteFunction
saveImages(img3)


# plt.subplot(111),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
