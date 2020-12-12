import numpy as np
import cv2
import os
from skimage.filters import threshold_otsu
from skimage import filters
from skimage import io

########################################################################################################################
Path = 'TrainingImages/'
images = os.listdir(Path)

for a in range(len(images)):
    image = cv2.imread(Path + images[a])
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.medianBlur(grayImage, 25)

    circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 120, param1=60, param2=30, minRadius=20, maxRadius=45)
    circles = np.uint16(np.around(circles))
    print("circular Hough transform complete...")

    for pt in circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(image, (a, b), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
        cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Circle", image)
        cv2.waitKey(0)
########################################################################################################################
# Path = 'TestingImages/'
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


