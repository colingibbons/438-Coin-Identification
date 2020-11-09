import cv2
import matplotlib.pyplot as plt
import matplotlib
from skimage.filters import threshold_otsu as Otsu
import numpy as np
from skimage.morphology import binary_closing, area_closing, closing, disk
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import scipy.ndimage as nd
from skimage import color

# read in images
image1 = cv2.imread('coinImage1.png')
image2 = cv2.imread('coinImage2.png')

# TODO consider switching to plt.imshow() so we don't have to use the wait key?
fig1 = plt.figure(figsize=(15, 15))
fig1.suptitle('First Coin Image', fontsize=30)
plt.imshow(image1)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
fig2 = plt.figure(figsize=(15, 15))
fig2.suptitle('Second Coin Image', fontsize=30)
plt.imshow(image2)
plt.xticks([]), plt.yticks([])
plt.show()

# display the images. press 0 key to close them
# cv2.imshow('First coin image', image1)
# cv2.imshow('Second coin image', image2)
# cv2.waitKey(0)

# generate grayscale images
grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Otsu Thresholding
thresh = Otsu(grayImage1)
binary = np.uint8(grayImage1 > thresh)
binaryNot = np.uint8(cv2.bitwise_not(binary))

# Morphological filtering
structElement = disk(radius=50)
closedBinaryImage = np.uint8(closing(binaryNot, structElement))

# remove artifacts connected to image border
cleared = clear_border(closedBinaryImage)

# label image regions
labelImage = label(cleared)

imgRows, imgCols, _ = image1.shape
objects = np.unique(labelImage)
numObjects = len(objects) - 1
objectSizeThres = 200
objectList = [] * numObjects

for a in range(1, numObjects+1):
    objectCoord = np.where(labelImage == a)
    objectRows = objectCoord[0]
    objectCols = objectCoord[1]
    if len(objectRows) and len(objectCols) > objectSizeThres:
        tempImage = np.isin(labelImage, a).astype(np.uint8)
        objectList.append(tempImage)

# for a in range(1, numObjects):
#     objectCoord = np.where(labelImage == a)
#     objectRows = objectCoord[0]
#     objectCols = objectCoord[1]
#     if len(objectRows) and len(objectCols) > objectSizeThres:
#         object = np.zeros((imgRows, imgCols), dtype='int')
#         for r in range(objectRows[0], objectRows[-1]):
#             for c in range(objectCols[0], objectCols[-1]):
#                 object[r, c] = 1
#         objectList[a-1] = object
# possible addition box-in segmented objects
# # to make the background transparent, pass the value of `bg_label`,
# # and leave `bg_color` as `None` and `kind` as `overlay`
# image_label_overlay = label2rgb(label_image, image=closedBinaryImage, bg_label=0)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(image_label_overlay)
#
# for region in regionprops(label_image):
#     # take regions with large enough areas
#     if region.area <= 100:
#         # draw rectangle around segmented coins
#         minr, minc, maxr, maxc = region.bbox
#         rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                   fill=False, edgecolor='red', linewidth=2)
#         ax.add_patch(rect)
#
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()


a = 0
