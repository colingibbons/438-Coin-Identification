import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC

# import segmentation functions housed in separate file
import coinSegFunctions as coins

# debugging toggle
debug = False

# Hough method toggle
Hough = True

# initializing training set
trainPath = 'TrainingImages/'
trainNames = os.listdir(trainPath)

# initializing feature vectors and training labels
trainingFeatures = []
trainingLabels = []

# determining location of coin type in trainNames
nickels = [2, 3]  # once more is included, should change to nickels = [0,1,2]
pennies = [4, 5]  # pennies = [3,4,5]
dimes = [0, 1]
quarters = [6, 7]  # etc


########################################################################################################################
# Creating List of all segmented coins for training input

penniesList = []
for a in range(len(pennies)):
    b = pennies[a]
    imagePath = trainPath + trainNames[b]
    penniesFromImage = coins.CoinSegmentation(imagePath, Hough)
    penniesList += penniesFromImage

nickelsList = []
for a in range(len(nickels)):
    b = nickels[a]
    imagePath = trainPath + trainNames[b]
    nickelsFromImage = coins.CoinSegmentation(imagePath, Hough)
    nickelsList += nickelsFromImage

quartersList = []
for a in range(len(quarters)):
    b = quarters[a]
    imagePath = trainPath + trainNames[b]
    quartersFromImage = coins.CoinSegmentation(imagePath, Hough)
    quartersList += quartersFromImage

dimesList = []
for a in range(len(dimes)):
    b = dimes[a]
    imagePath = trainPath + trainNames[b]
    dimesFromImage = coins.CoinSegmentation(imagePath, Hough)
    dimesList += dimesFromImage

########################################################################################################################
if debug:
    for a in range(len(penniesList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 700, 1000)
        cv2.imshow("Individual Segmentations", penniesList[a])
        cv2.waitKey(0)

    for a in range(len(nickelsList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 700, 1000)
        cv2.imshow("Individual Segmentations", nickelsList[a])
        cv2.waitKey(0)

    for a in range(len(dimesList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 700, 1000)
        cv2.imshow("Individual Segmentations", dimesList[a])
        cv2.waitKey(0)

    for a in range(len(quartersList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 700, 1000)
        cv2.imshow("Individual Segmentations", quartersList[a])
        cv2.waitKey(0)

########################################################################################################################
# loop over the training dataset
print('[STATUS] Started extracting Haralick textures..')

# begin texture feature extraction for each object list
trainingFeatures, trainingLabels = coins.Training(penniesList, 'penny', trainingFeatures, trainingLabels)
trainingFeatures, trainingLabels = coins.Training(nickelsList, 'nickel', trainingFeatures, trainingLabels)
trainingFeatures, trainingLabels = coins.Training(dimesList, 'dime', trainingFeatures, trainingLabels)
trainingFeatures, trainingLabels = coins.Training(quartersList, 'quarter', trainingFeatures, trainingLabels)

# have a look at the size of our feature vector and labels
print("Training features: {}".format(np.array(trainingFeatures).shape))
print("Training labels: {}".format(np.array(trainingLabels).shape))

# create the classifier
print("[STATUS] Creating the classifier..")
clf_svm = LinearSVC(random_state=9, dual=False)

# fit the training data and labels
print("[STATUS] Fitting data/label to model..")
clf_svm.fit(trainingFeatures, trainingLabels)

########################################################################################################################

# get test images from file
testPath = 'TestingImages/'
testNames = os.listdir(testPath)

# loop through test images, detect coins, and add them to a list
testCoinsList = []
for i in testNames:
    imagePath = testPath + i
    testCoinsFromImage = coins.CoinSegmentation(imagePath, Hough)
    if testCoinsFromImage:
        testCoinsList += testCoinsFromImage

# make predictions for each coin in the test set
prediction = coins.Testing(testCoinsList, clf_svm)


