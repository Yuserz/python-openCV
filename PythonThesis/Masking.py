# import the necessary packages
import numpy as np
import argparse
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="image/1.jpg",
	help="path to the input image")
args = vars(ap.parse_args())

# load the original input image and display it to our screen
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# a mask is the same size as our image, but has only two pixel
# values, 0 and 255 -- pixels with a value of 0 (background) are
# ignored in the original image while mask pixels with a value of
# 255 (foreground) are allowed to be kept
rec = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(rec, (10, 10), (160, 145), 255, -1)
cv2.imshow("Rectangular Mask", rec)

# draw a circle
circle = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(circle,( 85, 77), 70, 255, -1)
cv2.imshow("Circle mask", circle)

mask = cv2.bitwise_and(rec, circle)
cv2.imshow("mask", mask)

# apply our mask -- notice how only the person in the image is
# cropped out
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)