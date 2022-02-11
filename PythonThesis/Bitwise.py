import cv2
import numpy
import numpy as np


# draw a rectangle
rectangle = np.zeros((300, 300), np.uint8)
cv2.rectangle(rectangle,(25, 25), (275, 275), (255,255,255), -1)
cv2.imshow("Rectangle", rectangle)
#
# draw a circle
circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)


bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()


#
# img1 = np.zeros((250,500,3), np.uint8)
# img1 = cv2.rectangle(img1,(200,0), (300, 100),(255,255,255), -1)
# img2 = np.full((250, 500, 3), 255, dtype=np.uint8)
# img2 = cv2.rectangle(img2, (0, 0), (250, 250), (0, 0, 0), -1)
# #
# # bitAnd = cv2.bitwise_not(img2,img1)
# bitnot1 = cv2.bitwise_not(img2)
# bitnot2 = cv2.bitwise_not(img1)
#
# # cv2.imshow('img1',img1)
# # cv2.imshow('img2',img2)
# cv2.imshow('Bitwise1', bitnot1)
# cv2.imshow('Bitwise2', bitnot2)
#
#


