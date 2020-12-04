import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu as Otsu
import numpy as np
import glob
from skimage.morphology import disk
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, erosion
import os
import mahotas as mt
from sklearn.svm import LinearSVC
from itertools import combinations
from sklearn.neural_network import MLPClassifier

# debugging toggle
debug = False

# initializing training set
trainPath = 'C:/ECE 438 Final Project/438-Coin-Identification/TrainingImages'
trainNames = os.listdir(trainPath)

# initializing feature vectors and training labels
trainingFeatures = []
trainingLabels = []

# determining location of coin type in trainNames
nickels = [0]  # once more is included, should change to nickels = [0,1,2]
pennies = [1]  # pennies = [3,4,5]
quarters = [2]  # etc

########################################################################################################################
def CoinSegmentation(imageName):
    image = cv2.imread(imageName)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu Thresholding
    thresh = Otsu(grayImage)
    binary = np.uint8(grayImage > thresh)
    binaryNot = np.uint8(cv2.bitwise_not(binary))

    # Morphological filtering
    structElement = disk(radius=30)
    closedBinaryImage = np.uint8(closing(binaryNot, structElement))
    print("closing operation complete.")  # added this verification message because the operation takes a while

    cleared = clear_border(closedBinaryImage)

    # label image regions
    labelImage = label(cleared)

    imgRows, imgCols, _ = image.shape
    objects = np.unique(labelImage)
    numObjects = len(objects) - 1
    objectSizeThres = 200
    objectList = [] * numObjects

    for a in range(1, numObjects + 1):
        objectCoord = np.where(labelImage == a)
        objectRows = objectCoord[0]
        objectCols = objectCoord[1]
        if len(objectRows) and len(objectCols) > objectSizeThres:
            tempImage = np.isin(labelImage, a).astype(np.uint8)
            objectList.append(tempImage)

    extractedObjects = [cv2.bitwise_and(image, image, mask=(255 * objectList[k])) for k in range(len(objectList))]
    return extractedObjects


# function to extract haralick textures from an image
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean


# function to run through segmented coin list
def Training(ImageList, objectName, trainingFeatures, trainingLabels):
    for i in range(len(ImageList)):
        print("Processing Image - {} for {}".format(i + 1, objectName))
        # read the training image
        image = ImageList[i]

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray)

        # append the feature vector and label
        trainingFeatures.append(features)
        trainingLabels.append(objectName)
    print('\n')
    return trainingFeatures, trainingLabels


def Testing(inputImage, clf_svm):
    coinPrediction = []
    for x in range(len(inputImage)):
        # read the input image
        image = inputImage[x]

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray)

        # evaluate the model and predict label
        prediction = clf_svm.predict(features.reshape(1, -1))[0]

        # show the label
        cv2.putText(image, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        print("Prediction - {}".format(prediction))
        coinPrediction.append(prediction)

        # display the output image
        cv2.imshow("Test Image", image)
        cv2.waitKey(0)

    return coinPrediction
########################################################################################################################
# Creating List of all segmented coins for training input
for a in range(len(pennies)):
    b = pennies[a]
    penniesList = CoinSegmentation(trainNames[b])

for a in range(len(nickels)):
    b = nickels[a]
    nickelsList = CoinSegmentation(trainNames[b])

for a in range(len(quarters)):
    b = quarters[a]
    quartersList = CoinSegmentation(trainNames[b])

########################################################################################################################
if debug:
    for a in range(len(penniesList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 1000, 700)
        cv2.imshow("Individual Segmentations", penniesList[a])
        cv2.waitKey(0)

    for a in range(len(nickelsList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 1000, 700)
        cv2.imshow("Individual Segmentations", nickelsList[a])
        cv2.waitKey(0)

    for a in range(len(quartersList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 1000, 700)
        cv2.imshow("Individual Segmentations", quartersList[a])
        cv2.waitKey(0)

########################################################################################################################
# loop over the training dataset
print('[STATUS] Started extracting haralick textures..')

# begin texture feature extraction for each object list
trainingFeatures, trainingLabels = Training(penniesList, 'penny', trainingFeatures, trainingLabels)
trainingFeatures, trainingLabels = Training(nickelsList, 'nickel', trainingFeatures, trainingLabels)
trainingFeatures, trainingLabels = Training(quartersList, 'quarter', trainingFeatures, trainingLabels)

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

# loop over the test images
testPath = 'C:/ECE 438 Final Project/438-Coin-Identification/TestingImages'
testNames = os.listdir(testPath)

imageList1 = CoinSegmentation(testNames[0])
imageList2 = CoinSegmentation(testNames[1])

prediction = Testing(imageList1, clf_svm)
prediction = Testing(imageList2, clf_svm)


a = 0
