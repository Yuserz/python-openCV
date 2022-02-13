import cv2
import numpy as np
import utils

path = "1.jpg"
nailImg = "bl.png"

widthImage = 500
heightImg = 500

img = cv2.imread(path)

img = cv2.resize(img, (widthImage, heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 200, 200)

contours, hierarchy = cv2.findContours(
    imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

rectCont = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCont[0])
gradePoints = utils.getCornerPoints(rectCont[1])

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

imgBlank = np.zeros_like(img)

imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgBlank, imgBlank])
imageStacked = utils.stackImages(imageArray, 0.5)

cv2.imshow("original", imageStacked)
cv2.waitKey(0)
