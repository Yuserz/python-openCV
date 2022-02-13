import numpy as np
from matplotlib import pyplot as plt
import cv2

image = cv2.imread("testSample.jpg")  # load image

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR to HSV conversion
hsv_img = cv2.resize(hsv_img, (250, 250))

img_s = hsv_img[:, :, 1]  # Extracting Saturation channel on which we will work

img_s_blur = cv2.GaussianBlur(img_s, (7, 7), 0)  # smoothing before applying  threshold

canny = cv2.Canny(img_s_blur, 100, 200)

img_s_binary = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # Thresholding to generate binary image (ROI detection)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img_s_binary = cv2.morphologyEx(img_s_binary, cv2.MORPH_OPEN, kernel, iterations=3)  # reduce some noise

img_croped = cv2.bitwise_and(img_s, img_s_binary) * 5  # ROI only image extraction & contrast enhancement, you can crop this region 

abs_grad_x = cv2.convertScaleAbs(cv2.Sobel(img_croped, cv2.CV_64F, 1, 0, ksize=3))
abs_grad_y = cv2.convertScaleAbs(cv2.Sobel(img_croped, cv2.CV_64F, 0, 1, ksize=3))
grad = cv2.addWeighted(abs_grad_x, .5, abs_grad_y, .5, 0)  # Gradient calculation
grad = cv2.medianBlur(grad, 13)

edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contours Detection
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnt = None
max_area = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > max_area:  # Filtering contour
        max_area = area
        cnt = c

cv2.drawContours(hsv_img, [cnt], 0, (0, 255, 0), 3)

def showImages(imgs, titles):
    for i in range(0, 4):
        plt.subplot(141+i), plt.imshow(imgs[i], 'gray'), plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


images = [image, canny, img_croped, edges]
titles = ["Image", "Canny Edge", "Image Cropped", "Edges"]
showImages(images, titles)
