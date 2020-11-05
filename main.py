import cv2

# read in images
image1 = cv2.imread('coinImage1.png')
image2 = cv2.imread('coinImage2.png')

# display the images. press 0 key to close them
cv2.imshow('First coin image', image1)
cv2.imshow('Second coin image', image2)
cv2.waitKey(0)

# generate grayscale images
grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayIMage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
