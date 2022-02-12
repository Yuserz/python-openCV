import cv2 as cv
import cv2
import numpy as np
import math
import glob


def saveImages(img):
    # for naming contoured image
    num = 1

    for i in img:
        cv.imwrite('segImg/' + str(num) + '.jpg', i, (cv.IMWRITE_JPEG_QUALITY, 100))
        num += 1

    print("Saved Successfully!")

def largestContours(canny, img):
    # Finding Contour
    contours, _ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contoured_img = np.copy(img)  # Contours change original image.


    # Get all perimeter
    perimeter = []
    i = 0

    # Find perimeter for each contour i = id of contour
    for each_cnt in contours:
        prm = cv.arcLength(each_cnt, True)
        perimeter.append([prm, i])
        i += 1

    # Sort perimeter and return 1 array with 4 points only
    perimeter = quick_sort(perimeter)

    unified = []
    max_index = []
    # Draw all contours
    for i in range(len(contours)):
        index = perimeter[i][1]
        max_index.append(index)
        # cv.drawContours(contoured_img, contours, index, (0, 0, 255), 2)

    # Get convexhull for max contours and draw them
    conContour = np.vstack(contours[i] for i in max_index)
    hull = cv.convexHull(conContour)
    unified.append(hull)
    cv.drawContours(contoured_img, unified,-1, (0,255,0), 2)

    #Boundingbox will be the final perimmeter of the image
    boundingBoxes = [cv.boundingRect(c) for c in unified]
    print("BoundingBox:", boundingBoxes)

    return contoured_img, contours, perimeter, hull, unified, boundingBoxes

def quick_sort(p):
    if len(p) <= 1:
        return p

    pivot = p.pop(0)
    low, high = [], []
    for entry in p:
        if entry[0] > pivot[0]:
            high.append(entry)
        else:
            low.append(entry)
    return quick_sort(high) + [pivot] + quick_sort(low)

def grCut(image, bd, cx, cy, Radius):
    #rectangle that contains the object

    #Rectangle will get the 4 index in the boundingBox of the contour
    global rect
    for boundingBox in bd:
        rect = boundingBox

    #ADD 1 TO RECT ARRAY TO PREVENT ERROR WHEN HEIGHT OR WIDTH IS ZERO
    rect = tuple(np.array(rect)+1)

    #split the Perimeter of boundingBox
    coordinates = np.array_split(rect, 2)

    #store to point variable
    global pt1
    global pt2
    n = 0
    for c in coordinates:
        if n == 0:
            pt1 = c
            n +=1
        else:
            pt2 = c

    #Create 2 mask
    #Rectangular mask
    rec = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(rec,(pt1), (pt2), 255, -1)
    # cv2.imshow("Rectangular Mask", rec)


    # circle mask
    circle = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(circle, (cx, cy), int(Radius) - 10, 255, -1) # subtracted 10 to original radius to eliminate excess pixels
    # cv2.imshow("Circle mask", circle)

    #combined using bitwise_and operator
    mask = cv2.bitwise_and(rec, circle)
    # cv2.imshow("mask", mask)

    # apply our mask
    # cropped out
    masked = cv2.bitwise_and(image, image, mask)
    # cv2.imshow("Mask Applied to Image", masked)


    # Values needed for algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Grabcut
    cv2.grabCut(image, masked, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD),0,1).astype('uint8')
    img = image*mask2[:, : , np.newaxis]

    return img

def contourAnalysis(uni):

    # Contour Analysis
    global cX, cY, M
    for contour in uni:
        global M
        # Get the image moment for contour
        M = cv.moments(contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # set values as what you need in the situation
            cX, cY = 0, 0


    # Draw a circle to indicate the contour center
    cv.circle(contoured_img, (cX, cY), 5, (255, 0, 0), -1)

    # solving Area
    areaCon = int(M["m00"])

    print("\nArea", areaCon)

    # Solving the radius using area value
    pi = 3.14159
    area = areaCon

    radius = math.sqrt(area / pi)

    print("Radius", radius)

    return M, cX, cY, area, radius

# -----------------------------------------START---------------------------------------------------
#Variables

# images = []
imgList = []
origImg = []
grayImg = []
blurImg = []
cannyEdge = []
contourImg = []
gCut = []
convexHull = []
bBox= []
uni = []
cx = []
cy = []
Radius = []


#img folder Directory
imdir = 'image/'
ext = ['png', 'jpg'] # Add image formats here

#Locate image and extend
files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

#Read image
images = [cv2.imread(file) for file in files]

#make a list of original image
for a in images:
    origImg.append(a)

#convert image to gray
for i in origImg:
    # convert to grayscale
    gray = cv.cvtColor(i, cv.COLOR_RGB2GRAY)
    grayImg.append(gray)

#Blur image
for j in grayImg:
    # {GAUSSIANBLUR VALUE} kernel size is none negative & odd numbers only
    #SMOOTHING(Applying GaussianBlur)
    ks = 5
    sigma = 5
    blur = cv.GaussianBlur(j, (ks, ks),sigmaX=sigma, sigmaY=sigma)
    blurImg.append(blur)

#Process Canny
for k in blurImg:
    # CANNY(Finding Edge)
    canny = cv.Canny(k, 5, 70, L2gradient=True)
    cannyEdge.append(canny)

#Find Contour(Find & Draw)
for c, o in zip(cannyEdge, origImg):
    # FINDING CONTOUR
    contoured_img, contours, perimeters, hull, unified, boundingBoxes = largestContours(c, o)
    contourImg.append(contoured_img)
    convexHull.append(hull)
    uni.append(unified)
    bBox.append(boundingBoxes)
    # Box.append(boundingBoxes)

#Contour Analysis
for u in uni:
    M, cX, cY, area, radius = contourAnalysis(u)
    cx.append(cX)
    cy.append(cY)
    Radius.append(radius)


#GrabCut the contoured Nail
for h, g, x , y , r in zip(origImg, bBox, cx, cy ,Radius):
    #Cutting the contoured nail
    img = grCut(h, g, x , y ,r)
    gCut.append(img)

#WriteFunction
saveImages(gCut)
cv2.waitKey(0)
# saveImages(origImg)
# saveImages(cannyEdge)
# saveImages(contourImg)
# saveImages(cannyEdge)

