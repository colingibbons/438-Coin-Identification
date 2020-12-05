import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu as Otsu
import numpy as np
from skimage.morphology import disk
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
from itertools import combinations
from sklearn.neural_network import MLPClassifier

# dictionary containing the area ratios for each combination of coins.
coinRatios = {
    # These are the ratios of the smaller coins to the larger coins
    "Dime-Quarter": 0.504344,
    "Penny-Quarter": 0.714169,
    "Dime-Penny": 0.733127,
    "Nickel-Quarter": 0.737267,
    "Dime-Nickel": 0.774993,
    "Penny-Nickel": 0.900731,

    # These are the ratios of the larger coins to the smaller coins (the reciprocals of the above values)
    "Nickel-Penny": 1.110209,
    "Nickel-Dime": 1.290334,
    "Quarter-Nickel": 1.356360,
    "Penny-Dime": 1.364020,
    "Quarter-Penny": 1.400229,
    "Quarter-Dime": 1.982773
}

# read in images
image1 = cv2.imread('TestingImages/testcoins.jpg')
image2 = cv2.imread('pennies.jpg')
image3 = cv2.imread('quarters.png')
image4 = cv2.imread('nickels.png')

# TODO consider switching to plt.imshow() so we don't have to use the wait key?
# fig1 = plt.figure(figsize=(15, 15))
# fig1.suptitle('First Coin Image', fontsize=30)
# plt.imshow(image1)
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# fig2 = plt.figure(figsize=(15, 15))
# fig2.suptitle('Second Coin Image', fontsize=30)
# plt.imshow(image2)
# plt.xticks([]), plt.yticks([])
# plt.show()

# img = cv2.medianBlur(image1, 5)
# cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100,
#                             param1=50, param2=30, minRadius=20, maxRadius=99)
#
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     cv2.circle(cimg, (i[0],i[1]), i[2], (0,255,0), 2)

#cv2.imshow('detected circles', cimg)
#cv2.waitKey(0)

# generate grayscale images
grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

### processing for image 1 ###

# Otsu Thresholding
thresh = Otsu(grayImage1)
binary = np.uint8(grayImage1 > thresh)
binaryNot = np.uint8(cv2.bitwise_not(binary))

# Morphological filtering
structElement = disk(radius=30)
closedBinaryImage = np.uint8(closing(binaryNot, structElement))
print("closing operation complete.")  # added this verification message because the operation takes a while

# remove artifacts connected to image border
# TODO find another way to do this - removes the coin objects themselves if no actual artifacts present
cleared = clear_border(closedBinaryImage)

# label image regions
labelImage = label(cleared)

imgRows, imgCols, _ = image1.shape
objects = np.unique(labelImage)
numObjects = len(objects) - 1
objectSizeThres = 200
objectList1 = [] * numObjects

for a in range(1, numObjects+1):
    objectCoord = np.where(labelImage == a)
    objectRows = objectCoord[0]
    objectCols = objectCoord[1]
    if len(objectRows) and len(objectCols) > objectSizeThres:
        tempImage = np.isin(labelImage, a).astype(np.uint8)
        objectList1.append(tempImage)

### processing for image 2 ####
thresh = Otsu(grayImage2)
binary = np.uint8(grayImage2 > thresh)
binaryNot = np.uint8(cv2.bitwise_not(binary))

# Morphological filtering
structElement = disk(radius=30)
closedBinaryImage = np.uint8(closing(binaryNot, structElement))
print("closing operation complete.")  # added this verification message because the operation takes a while

# remove artifacts connected to image border
# TODO find another way to do this - removes the coin objects themselves if no actual artifacts present
cleared = clear_border(closedBinaryImage)

# label image regions
labelImage = label(cleared)

imgRows, imgCols, _ = image2.shape
objects = np.unique(labelImage)
numObjects = len(objects) - 1
objectSizeThres = 200
objectList2 = [] * numObjects

for a in range(1, numObjects+1):
    objectCoord = np.where(labelImage == a)
    objectRows = objectCoord[0]
    objectCols = objectCoord[1]
    if len(objectRows) and len(objectCols) > objectSizeThres:
        tempImage = np.isin(labelImage, a).astype(np.uint8)
        objectList2.append(tempImage)

# colors = ('b', 'g', 'r')
# for j in range(len(objectList)):
#     plt.figure()
#     for i,col in enumerate(colors):
#         hist = cv2.calcHist([image1], [i], objectList[j], [256], [0,256])
#         plt.plot(hist,color=col)
#         plt.xlim([0,256])
#     plt.show()

# generate list of areas of each object for use in computing coin size ratios
# objectAreas = [np.count_nonzero(objectList[i]) for i in range(len(objectList))]

# generate a list of object area pairs and compute ratios for each pair
# objectCombos = list(combinations(objectAreas, 2))
# objectRatios = [objectCombos[j][0] / objectCombos[j][1] for j in range(len(objectCombos))]

# get individual coins from each image by ANDing original image with mask
extractedObjects1 = [cv2.bitwise_and(image1, image1, mask=(255*objectList1[k])) for k in range(len(objectList1))]
extractedObjects2 = [cv2.bitwise_and(image2, image2, mask=(255*objectList2[k])) for k in range(len(objectList2))]

SE = disk(10)
coinMask1 = cv2.erode(objectList1[2], SE)
coinMask2 = cv2.erode(objectList2[2], SE)

# Do feature extraction on object from first image
brisk = cv2.BRISK_create()
gray = cv2.cvtColor(extractedObjects1[2], cv2.COLOR_RGB2GRAY)
(kps, descs) = brisk.detectAndCompute(gray, mask=coinMask1)

# match features of object in first image to features in each object from second image, and display
for l in range(len(extractedObjects2)):
    gray2 = cv2.cvtColor(extractedObjects2[l], cv2.COLOR_RGB2GRAY)
    (kps2, descs2) = brisk.detectAndCompute(gray2, mask=coinMask2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs, descs2)
    output = cv2.drawMatches(extractedObjects1[2], kps, extractedObjects2[l], kps2, matches[:10], None)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 1200, 900)
    cv2.imshow("output", output)
    cv2.waitKey(0)

a = 0
