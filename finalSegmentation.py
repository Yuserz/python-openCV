import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('testSample.jpg', 0)

cv2.imshow('final', img)
cv2.waitKey(0)
