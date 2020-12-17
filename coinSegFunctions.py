import cv2
import numpy as np
import mahotas as mt
from mahotas.features import surf

########################################################################################################################
# Segmenting coins with circular Hough Transform & storing shape features (Area & Perimeter)
def CoinSegmentation(imageName, Hough, plot):
    image = cv2.imread(imageName)
    X, Y, _ = image.shape
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.medianBlur(grayImage, 25)

    if Hough:
        # Testing Circular Hough Transform
        circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 165, param1=50, param2=20, minRadius=150, maxRadius=350)
        if circles is None:
            return None
        circles = np.uint16(np.around(circles))
        print("circular Hough transform complete...")

        shapeFeatures = []
        coinList = []
        for pt in circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # create a mask to extract coin from original image
            circleMask = np.zeros(image.shape, dtype=np.uint8)
            cv2.circle(circleMask, (a, b), r, (255, 255, 255), cv2.FILLED)
            coinList.append(image & circleMask)

            # Extracting shape features from masks (Area & Perimeter)
            grayCircleMask = cv2.cvtColor(circleMask, cv2.COLOR_BGR2GRAY)
            area = np.sum(grayCircleMask/255)
            grayCircleMask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(grayCircleMask, (a, b), r, (255, 255, 255), 1)
            perimeter = np.sum(grayCircleMask/255)
            shapeFeatures.append([area, perimeter])

            # plot toggle for checking Hough's outputs
            if plot:
                # Draw the circumference of the circle.
                cv2.circle(image, (a, b), r, (0, 255, 0), 2)
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
                cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Detected Circle", 600, 800)
                cv2.imshow("Detected Circle", image)
                cv2.waitKey(0)

        return coinList, shapeFeatures

########################################################################################################################
# function to extract Haralick textures from an image
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image, ignore_zeros=True, distance=27)

    # take the mean of each of the four gray-level co-occurrence matrices and return the result
    ht_mean = textures.mean(axis=0)

    return ht_mean

########################################################################################################################
# Runs through training segmented images and call extract features function for Haralick textures
# returns texture features and training labels
def Training(ImageList, objectName, reduceFeatures):

    trainingFeatures = []
    trainingLabels = []
    for i in range(len(ImageList)):
        # print("Processing Image - {} for {}".format(i + 1, objectName))
        # read the training image
        image = ImageList[i]

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray)

        # remove x amount of features from the end of collected features (determined by reduceFeatures)
        if reduceFeatures is not None:
            features = features[:reduceFeatures]
        trainingFeatures.append(features)

        # append the feature vector and label
        trainingLabels.append(objectName)

    return trainingFeatures, trainingLabels

########################################################################################################################
# Extracting features from testing segmentations and making prediction based off of the model
def Testing(inputImage, clf_svm, correctString, testCoinsShapes, plot, reduceFeatures):

    coinPrediction = []
    count = 0
    for a in range(len(inputImage)):
        # read the input image
        image = inputImage[a]
        X, Y, _ = image.shape

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray)
        if reduceFeatures is not None:
            features = features[:reduceFeatures]

        # Addition of testing images shape features
        features = np.insert(features, 0, testCoinsShapes[a])

        # evaluate the model and predict label
        prediction = clf_svm.predict(features.reshape(1, -1))[0]

        # storing prediction
        coinPrediction.append(prediction)

        # plot toggle for viewing segmented test coin with prediction in upper left corner
        if plot:
            cv2.putText(image, prediction, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
            cv2.namedWindow("Test Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test Image", X, Y)
            cv2.imshow("Test Image", image)
            cv2.waitKey(0)

        # Comparing the prediction with the string of correct coin identities
        correct = correctString[a]
        testOutcome = correct == prediction
        if testOutcome:
            count += 1

    # overall success rate calculation
    successRate = 100 * (count / len(inputImage))
    print("Success Rate: {} %".format(successRate))

    return coinPrediction