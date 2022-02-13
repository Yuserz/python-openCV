import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# def quick_sort(p):
#     if len(p) <= 1:
#         return p
#
#     pivot = p.pop(0)
#     low, high = [], []
#     for entry in p:
#         if entry[0] > pivot[0]:
#             high.append(entry)
#         else:
#             low.append(entry)
#     return quick_sort(high) + [pivot] + quick_sort(low)



def showImages(imgs, titles):
    for i in range(0, 4):
        plt.subplot(141+i), plt.imshow(imgs[i], 'gray'), plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()



img = cv.imread('image/sample.jpg') #readimage

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

#Smoothing
#kernel size is none negative & odd numbers only
ks_width = 7
ks_height = 7
sigma_x = 50
sigma_y = 40
dst = None

img_blur = cv.GaussianBlur(gray,(ks_width, ks_height),sigma_x,dst,sigma_y)
# img_blur = cv.blur(gray,(0,0),0)#Smoothing
# Canny(Finding Edge)
canny = cv.Canny(img_blur,10,30,L2gradient=True)# Noise Reduction, Finding Intensity Gradient of the Image,

# Finding Contour
contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# img_contour = np.copy(img)  # Contours change original image.
# cv2.drawContours(img_contour, contours, -1, (0,255,0), 3) # Draw all - For visualization only

# # Contours -  maybe the largest perimeters pinpoint to the leaf?
# perimeter = []
# max_perim = [0, 0]
# i = 0
#
# # Find perimeter for each contour i = id of contour
# for each_cnt in contours:
#     prm = cv.arcLength(each_cnt, False)
#     perimeter.append([prm, i])
#     i += 1
#
# # Sort them
# perimeter = quick_sort(perimeter)
#
# unified = []
# max_index = []
# # Draw max contours
# for i in range(0, 3):
#     index = perimeter[i][1]
#     max_index.append(index)
for contour in contours:
    approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
    cv.drawContours(img, [approx], 0, (255, 0, 0), 3)
    # x = approx.ravel()[0]
    # y = approx.ravel()[1]

# # Get convex hull for max contours and draw them
# cont = np.vstack(contours[i] for i in max_index)
# hull = cv.convexHull(cont)
# unified.append(hull)
# cv.drawContours(img_contour, unified, -1,(0, 0, 255), 3)


# cv2.imwrite('contours_none_image1.jpg', image_copy)
# cv2.destroyAllWindows()



# Show images
images = [gray, img_blur, canny, img]
titles = ["Binary_img","Blurred_img", "Canny_edge", "Contour"]

showImages(images, titles)