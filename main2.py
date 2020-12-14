import cv2
import numpy as np
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

# Classifier Type
classifierType = 'K Nearest Neighbor'

# initializing training set
trainPath = 'TrainingImages2/'
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

# # data preprocessing
# trainingFeatures = preprocessing.normalize(trainingFeatures)

# have a look at the size of our feature vector and labels
print("Training features: {}".format(np.array(trainingFeatures).shape[1]))
print("Training labels: {}".format(np.array(trainingLabels).shape[0]))

# Try Leave One Out method on the training set and
correct = 0
for test_idx in range(len(trainingFeatures)):
    # create new instance of classifier
    clf_svm = LinearSVC(random_state=13, dual=False)
    # copy training features and labels
    newTraining = trainingFeatures.copy()
    newLabels = trainingLabels.copy()
    # remove the test sample from each list
    newTraining = pandas.DataFrame(newTraining)
    newTraining = newTraining.drop(newTraining.index[test_idx])
    # newTraining.pop(test_idx)
    newLabels.pop(test_idx)
    # fit the model using the remaining samples
    clf_svm.fit(newTraining, newLabels)
    # make a prediction, compare it to the actual label, and increment the counter if there's a match
    prediction = clf_svm.predict(trainingFeatures[test_idx].reshape(1, -1))[0]
    if prediction == trainingLabels[test_idx]:
        correct += 1

# print result of Leave One Out test
percent = 100 * (correct / len(trainingFeatures))
print("This model correctly predicted {}% of the samples".format(percent))

#TODO (if needed) Keep adding classsifiers from
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

#TODO this scaler addition is recommened but gives bad results after implenting
scaler = StandardScaler()
train_data = scaler.fit_transform(trainingFeatures)
print("[STATUS] Creating the classifier..")

if classifierType == 'SVC':
    # SVC classifier
    scaler = StandardScaler()
    train_data = scaler.fit_transform(trainingFeatures)
    classifier = SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
elif classifierType == 'Linear SVC':
    # Linear SVC classifier
    scaler = StandardScaler()
    train_data = scaler.fit_transform(trainingFeatures)
    classifier = LinearSVC(random_state=13, dual=False)
elif classifierType == 'Neural Network':
    # Neural Network
    PCT = PCA(n_components=13)
    PCT.fit(trainingFeatures)
    trainingFeatures = PCT.singular_values_
    classifier = MLPClassifier(learning_rate='adaptive',  max_iter=1000)
elif classifierType == 'K Nearest Neighbor':
    # Kth nearest neighbor
    classifier = KNeighborsClassifier(10)


# fit classifier with training data
print("[STATUS] Fitting data/label to model..")
classifier = classifier.fit(trainingFeatures, trainingLabels)


# # create the classifier
# print("[STATUS] Creating the classifier..")
# clf_svm = LinearSVC(random_state=13, dual=False)
#
# # fit the training data and labels
# print("[STATUS] Fitting data/label to model..")
# clf_svm.fit(trainingFeatures, trainingLabels)

########################################################################################################################

# get test images from file
testPath = 'TestingImages2/'
testNames = os.listdir(testPath)

# loop through test images, detect coins, and add them to a list
testCoinsList = []
for i in testNames:
    imagePath = testPath + i
    testCoinsFromImage = coins.CoinSegmentation(imagePath, Hough)
    if testCoinsFromImage:
        testCoinsList += testCoinsFromImage

# make predictions for each coin in the test set
prediction = coins.Testing(testCoinsList, classifier)
#prediction = coins.Testing(testCoinsList, clf_svm)


