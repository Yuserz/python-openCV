import cv2
import numpy as np


def filtering(img_gray, esp):
    if esp == "median":
        return cv2.medianBlur(img_gray, 5)
    elif esp == "gaussian":
        return cv2.GaussianBlur(img_gray, (5, 5), 0)
    elif esp == "bilateral":
        return cv2.bilateralFilter(img_gray, 5, 50, 100)


image = cv2.imread("testSample.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

# smoothing the image
image_blur = filtering(image_gray, "gaussian")
canny_edge = cv2.Canny(image_gray, 100, 200)

cv2.imshow("original", canny_edge)
cv2.waitKey(0)
