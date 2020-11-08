import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu as Otsu
import numpy as np
from skimage.morphology import binary_closing, area_closing, closing, disk
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

# object segmentation by using findcontours function
contours, _ = cv2.findContours(closedBinaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

a = 0
