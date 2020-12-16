import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# import segmentation functions housed in separate file
import coinSegFunctions as coins

# debugging toggle
debug = False

# Hough method toggle
Hough = True

# Shape features toggle
shape = True

# Classifier Type
classifierType = 'Linear SVC'

# initializing correct coin string
d = 'dime'
q = 'quarter'
n = 'nickel'
p = 'penny'

validationCRTstring = [q, d, p, q, q, q, d, p, q, p, n, p, d, q, n, n, n, p, d, q, p, p, d, d, q, d, q, n, q, n, p, q,
                       n, n, d, q, d, q, p, d, p, p, d, q, p, q, n, d, p, p, q, q, d, n, p, p, p, d, n, n, p, p, d, d]

# initializing training set
trainPath = 'TrainingImages3/'
trainNames = os.listdir(trainPath)

# initializing feature vectors and training labels
trainingShapeFeatures = []
trainingShapeLabels = []

# determining location of coin type in trainNames
nickels = [4, 5, 6, 7]  # once more is included, should change to nickels = [0,1,2]
pennies = [8, 9, 10, 11]  # pennies = [3,4,5]
dimes = [0, 1, 2, 3]
quarters = [12, 13, 14, 15]  # etc


########################################################################################################################
# Creating List of all segmented coins for training input
print('\n')
print("processing training image set...")

penniesList = []
pennyMasks = []
for a in range(len(pennies)):
    b = pennies[a]
    imagePath = trainPath + trainNames[b]
    penniesFromImage, masks = coins.CoinSegmentation(imagePath, Hough, plot=False)
    penniesList += penniesFromImage
    pennyMasks += masks

nickelsList = []
nickelMasks = []
for a in range(len(nickels)):
    b = nickels[a]
    imagePath = trainPath + trainNames[b]
    nickelsFromImage, masks = coins.CoinSegmentation(imagePath, Hough, plot=False)
    nickelsList += nickelsFromImage
    nickelMasks += masks

quartersList = []
quarterMasks = []
for a in range(len(quarters)):
    b = quarters[a]
    imagePath = trainPath + trainNames[b]
    quartersFromImage, masks = coins.CoinSegmentation(imagePath, Hough, plot=False)
    quartersList += quartersFromImage
    quarterMasks += masks

dimesList = []
dimeMasks = []
for a in range(len(dimes)):
    b = dimes[a]
    imagePath = trainPath + trainNames[b]
    dimesFromImage, masks = coins.CoinSegmentation(imagePath, Hough, plot=False)
    dimesList += dimesFromImage
    dimeMasks += masks
########################################################################################################################
print('\n')
print("processing validation image set...")
# get test images from file
testPath = 'ValidationSet/'
testNames = os.listdir(testPath)

# loop through test images, detect coins, and add them to a list
testCoinsList = []
testCoinsMasks = []
for i in testNames:
    imagePath = testPath + i
    testCoinsFromImage, masks = coins.CoinSegmentation(imagePath, Hough, plot=False)
    if testCoinsFromImage:
        testCoinsList += testCoinsFromImage
        testCoinsMasks += masks

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
print('\n')
print('[STATUS] Started extracting Haralick textures..')
successRate = []
featureSize = [10]
for x in featureSize:
    # initializing feature vectors and training labels
    trainingFeatures = []
    trainingLabels = []
    # begin texture feature extraction for each object list
    trainingFeatures, trainingLabels = coins.Training(penniesList, 'penny', trainingFeatures, trainingLabels, reduceFeatures = x, shape=shape, masks=pennyMasks)
    trainingFeatures, trainingLabels = coins.Training(nickelsList, 'nickel', trainingFeatures, trainingLabels, reduceFeatures = x, shape=shape, masks=nickelMasks)
    trainingFeatures, trainingLabels = coins.Training(dimesList, 'dime', trainingFeatures, trainingLabels, reduceFeatures = x, shape=shape, masks=dimeMasks)
    trainingFeatures, trainingLabels = coins.Training(quartersList, 'quarter', trainingFeatures, trainingLabels, reduceFeatures = x, shape=shape, masks=quarterMasks)


    # have a look at the size of our feature vector and labels
    # print("Training features: {}".format(np.array(trainingFeatures).shape[1]))
    # print("Training labels: {}".format(np.array(trainingLabels).shape[0]))

    #TODO (if needed) Keep adding classsifiers from
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

    print("[STATUS] Creating the classifier..")

    if classifierType == 'SVC':
        # SVC classifier
        # scaler = StandardScaler()
        # train_data = scaler.fit_transform(trainingFeatures)
        classifier = SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False)
    elif classifierType == 'Linear SVC':
        # Linear SVC classifier
        # scaler = StandardScaler()
        # train_data = scaler.fit_transform(trainingFeatures)
        classifier = LinearSVC(random_state=x, dual=False, fit_intercept=True)
    elif classifierType == 'Neural Network':
        # Neural Network
        PCT = PCA(n_components=13)
        PCT.fit(trainingFeatures)
        classifier = MLPClassifier(learning_rate='adaptive',  max_iter=1000)
    elif classifierType == 'K Nearest Neighbor':
        # Kth nearest neighbor
        classifier = KNeighborsClassifier(10)

    # fit classifier with training data
    print("[STATUS] Fitting data/label to model..")
    # classifier = classifier.fit(PCT.singular_values_ , trainingLabels)
    # classifier = classifier.fit(train_data, trainingLabels)
    classifier = classifier.fit(trainingFeatures, trainingLabels)


    ########################################################################################################################
    # make predictions for each coin in the test set

    prediction, success = coins.Testing(testCoinsList, classifier, validationCRTstring, plot=False, reduceFeatures=x, shape=shape, masks=testCoinsMasks)
    successRate.append(success)

minimumFeatures = np.min(featureSize)
for a in successRate:
    print("Success Rate: {} % with {} features".format(a, minimumFeatures+2))
    minimumFeatures += 1


