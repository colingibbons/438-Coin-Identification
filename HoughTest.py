import numpy as np
import cv2
import os
from skimage.draw import circle
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from scipy import ndimage
from skimage import filters
from skimage import io
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import matplotlib.pyplot as plt

trying = True

########################################################################################################################
Path = 'ValidationSet/'
images = os.listdir(Path)

for a in range(len(images)):
    image = cv2.imread(Path + images[a])
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.medianBlur(grayImage, 25)
    # thresh = threshold_otsu(grayImage)
    # binary = np.uint8(grayImage > thresh)
    # binaryNot = np.uint8(cv2.bitwise_not(binary))
    # binaryNot[binaryNot < 255] = 0
    #
    # # Contour find & plot
    # contours, hierarchy = cv2.findContours(binaryNot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros(image.shape, np.uint8)
    # A = cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # A = np.uint8(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY))
    # print("closing operation in progress...")
    # structElement = disk(radius=50)
    # closedBinaryImage = np.uint8(ndimage.binary_closing(A, structElement))
    # print("closing operation complete.")

    circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 165, param1=50, param2=20, minRadius=150, maxRadius=350)
    circles.shape[1]
    circles = np.uint16(np.around(circles))
    print("circular Hough transform complete...")

    coinList=[]
    for pt in circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        # create a mask to extract coin from original image
        rr, cc = circle(a, b, r)
        circleMask = np.zeros(image.shape)
        circleMask[cc, rr] = 255
        circleMask = np.uint8(circleMask)
        coinList.append(image & circleMask)
        # # Draw the circumference of the circle.
        # cv2.circle(image, (a, b), r, (0, 255, 0), 3)
        #
        # # Draw a small circle (of radius 1) to show the center.
        # cv2.circle(image, (a, b), 10, (0, 0, 255), cv2.FILLED)
        # cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
        # cv2.imshow("Detected Circle", image)
        # cv2.waitKey(0)

        a = 0
    # if trying:
    #     mask = np.zeros((image.shape), dtype=np.uint8)
    #     for pt in circles[0, :]:
    #         a, b, r = pt[0], pt[1], pt[2]
    #
    #         # Draw the circle.
    #         cv2.circle(mask, (a, b), r, (0, 255, 0), cv2.FILLED)
    #     mask = np.uint8(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    # circleImage = np.uint8(mask)
########################################################################################################################
Path = 'TestingImages/'
# images = os.listdir(Path)
#
# for a in range(len(images)):
#     image = cv2.imread(Path + images[a])
#     grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 120, param1=60, param2=30, minRadius=20, maxRadius=45)
#     circles = np.uint16(np.around(circles))
#     print("circular Hough transform complete...")
#
#     for pt in circles[0, :]:
#         a, b, r = pt[0], pt[1], pt[2]
#
#         # Draw the circumference of the circle.
#         cv2.circle(image, (a, b), r, (0, 255, 0), 2)
#
#         # Draw a small circle (of radius 1) to show the center.
#         cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
#         cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
#         cv2.imshow("Detected Circle", image)
#         cv2.waitKey(0)


