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
from itertools import combinations
from skimage.color import label2rgb
import scipy.ndimage as nd
from skimage import color

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
print("closing operation complete.")  # added this verification message because the operation takes a while

# remove artifacts connected to image border
# TODO find another way to do this - removes the coin objects themselves if no actual artifacts present
cleared = clear_border(closedBinaryImage)

# label image regions
labelImage = label(cleared)

plt.imshow(labelImage)
plt.show()

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

# generate list of areas of each object for use in computing coin size ratios
objectAreas = [np.count_nonzero(objectList[i]) for i in range(len(objectList))]

# generate a list of object area pairs and compute ratios for each pair
objectCombos = list(combinations(objectAreas, 2))
objectRatios = [objectCombos[j][0] / objectCombos[j][1] for j in range(len(objectCombos))]

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
