import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# import segmentation functions housed in separate file
import coinSegFunctions as coins

# checks the correct testing string
checkTest = False

# debugging toggle
debug = False

# Hough method toggle
Hough = True

# initializing correct coin string
d = 'dime'
q = 'quarter'
n = 'nickel'
p = 'penny'

validationCRTstring = [q, d, p, q, q, q, d, p, q, p, n, p, d, q, n, n, n, p, d, q, p, p, d, d, q, d, q, n, q, n, p, q,
                       n, n, d, q, d, q, p, d, p, p, d, q, p, q, n, d, p, p, q, q, d, n, p, p, p, d, n, n, p, p, d, d]

testCRTstring = [q, p, n, q, p, q, d, p, d, p, q, d, d, n, p, d, n, d, d, n, d, p, p, p, d, p, n, q, n, d,
                 q, p, q, n, d, d, p, q, n, n, q, p, q, n, q, q, q, n, d, q, d, p, q, d, q, p, n, q, n, p, p, n, q, d]


# initializing training set
trainPath = 'TrainingImages3/'
trainNames = os.listdir(trainPath)

# initializing feature vectors and training labels
trainingShapeFeatures = []
trainingShapeLabels = []

# determining location of coin type in trainNames
nickels = [4, 5, 6, 7]
pennies = [8, 9, 10, 11]
dimes = [0, 1, 2, 3]
quarters = [12, 13, 14, 15]


########################################################################################################################
# Creating segmented coin images from  training image set
print('\n')
print("processing training image set...")

penniesList = []
pennyShapes = []
for a in range(len(pennies)):
    b = pennies[a]
    imagePath = trainPath + trainNames[b]
    penniesFromImage, shapeFeatures = coins.CoinSegmentation(imagePath, Hough, plot=False)
    penniesList += penniesFromImage
    pennyShapes += shapeFeatures

nickelsList = []
nickelShapes = []
for a in range(len(nickels)):
    b = nickels[a]
    imagePath = trainPath + trainNames[b]
    nickelsFromImage, shapeFeatures = coins.CoinSegmentation(imagePath, Hough, plot=False)
    nickelsList += nickelsFromImage
    nickelShapes += shapeFeatures

quartersList = []
quarterShapes = []
for a in range(len(quarters)):
    b = quarters[a]
    imagePath = trainPath + trainNames[b]
    quartersFromImage, shapeFeatures = coins.CoinSegmentation(imagePath, Hough, plot=False)
    quartersList += quartersFromImage
    quarterShapes += shapeFeatures

    # remove last two false segmentations
    quartersList = quartersList[:36]
    quarterShapes = quarterShapes[:36]

dimesList = []
dimeShapes = []
for a in range(len(dimes)):
    b = dimes[a]
    imagePath = trainPath + trainNames[b]
    dimesFromImage, shapeFeatures = coins.CoinSegmentation(imagePath, Hough, plot=False)
    dimesList += dimesFromImage
    dimeShapes += shapeFeatures
########################################################################################################################
# Creating segmented coin images from  training image set
print('\n')
print("processing validation image set...")
# get test images from file
testPath = 'testingImages3/'
testNames = os.listdir(testPath)

testCoinsList = []
testCoinsShapes = []
for i in testNames:
    imagePath = testPath + i
    testCoinsFromImage, shapeFeatures = coins.CoinSegmentation(imagePath, Hough, plot=False)
    if testCoinsFromImage:
        testCoinsList += testCoinsFromImage
        testCoinsShapes += shapeFeatures

# verify that the string of correct coins is correct
if checkTest:
    for a in range(len(testCoinsList)):
        cv2.namedWindow("Individual Segmentations", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Individual Segmentations", 700, 1000)
        cv2.putText(testCoinsList[a], testCRTstring[a], (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 5)
        cv2.imshow("Individual Segmentations", testCoinsList[a])
        cv2.waitKey(0)

########################################################################################################################
# check segmentation of training images
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
# loop over segmented coins from training set
print('\n')
print('[STATUS] Started extracting Haralick textures..')
successRate = []

# begin texture feature extraction for each object list
# penny extraction & shape Features addition
print("penny feature extractions in progress...")
pennyFeatures, pennyLabels = coins.Training(penniesList, 'penny', reduceFeatures = 10)
pennyFeatures = [np.insert(pennyFeatures[a], 0, pennyShapes[a]) for a in range(len(pennyFeatures))]

# nickel extraction & shape Features addition
print("nickel feature extractions in progress...")
nickelFeatures, nickelLabels = coins.Training(nickelsList, 'nickel',  reduceFeatures = 10)
nickelFeatures = [np.insert(nickelFeatures[a], 0, nickelShapes[a]) for a in range(len(nickelFeatures))]

# dime extraction & shape Features addition
print("dime feature extractions in progress...")
dimeFeatures, dimeLabels = coins.Training(dimesList, 'dime', reduceFeatures = 10)
dimeFeatures = [np.insert(dimeFeatures[a], 0, dimeShapes[a]) for a in range(len(dimeFeatures))]

# quarter extraction & shape Features addition
print("quarter feature extractions in progress...")
quarterFeatures, quarterLabels = coins.Training(quartersList, 'quarter', reduceFeatures = 10)
quarterFeatures = [np.insert(quarterFeatures[a], 0, quarterShapes[a]) for a in range(len(quarterFeatures))]

# combining features and labels
trainingFeatures = []
trainingLabels = []
trainingFeatures.extend(pennyFeatures), trainingLabels.extend(pennyLabels)
trainingFeatures.extend(nickelFeatures), trainingLabels.extend(nickelLabels)
trainingFeatures.extend(dimeFeatures), trainingLabels.extend(dimeLabels)
trainingFeatures.extend(quarterFeatures), trainingLabels.extend(quarterLabels)

########################################################################################################################
# create various classifiers
print("[STATUS] Creating the classifier..")
classList = ['QDA', 'Linear SVC', 'K Nearest Neighbor', 'Random Forest']
for x in classList:
    classifierType = x
    if classifierType == 'SVC':
        # SVC classifier
        # scaler = StandardScaler()
        # train_data = scaler.fit_transform(trainingFeatures)
        classifier = SVC(C=0.0001, kernel='linear')
    elif classifierType == 'Linear SVC':
        # Linear SVC classifier
        # scaler = StandardScaler()
        # train_data = scaler.fit_transform(trainingFeatures)
        classifier = LinearSVC(C=0.0001, random_state=12, dual=False, fit_intercept=False)
    elif classifierType == 'QDA':
        # Neural Network
        # PCT = PCA(n_components=12)
        # PCT.fit(trainingFeatures)
        # classifier = MLPClassifier(activation='logistic', max_iter=5000)
        classifier = QuadraticDiscriminantAnalysis(reg_param=0.01)
    elif classifierType == 'K Nearest Neighbor':
        # Kth nearest neighbor
        classifier = KNeighborsClassifier(15)

    elif classifierType == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=1000, max_features='sqrt', class_weight='balanced_subsample')

    # Createing model by fitting classifier with training data
    print("[STATUS] Fitting data/label to model..")
    classifier = classifier.fit(trainingFeatures, trainingLabels)

    # make predictions for each coin in the test set
    print("Processing/predicting validation set...")
    prediction = coins.Testing(testCoinsList, classifier, testCRTstring, testCoinsShapes,  plot=False, reduceFeatures=10)



