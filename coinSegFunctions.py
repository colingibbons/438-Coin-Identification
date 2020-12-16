import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.draw import circle
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
import mahotas as mt
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from skimage import filters
from mahotas.features import surf

def CoinSegmentation(imageName, Hough, plot):
    image = cv2.imread(imageName)
    X, Y, _ = image.shape
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.medianBlur(grayImage, 25)
    maskList = []

    if Hough:
        # Testing Circular Hough Transform
        # circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 120, param1=60, param2=30, minRadius=20, maxRadius=45) # Hough settins for original images
        circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 165, param1=50, param2=20, minRadius=150, maxRadius=350)   # Hough settings for new images
        if circles is None:
            return None
        circles = np.uint16(np.around(circles))
        print("circular Hough transform complete...")

        coinList = []
        allCircles = np.zeros(image.shape, dtype=np.uint8)
        for pt in circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # create a mask to extract coin from original image
            rr, cc = circle(a, b, r)
            circleMask = np.zeros(image.shape)
            circleMask[cc, rr] = 255
            allCircles[cc, rr] = 255
            circleMask = np.uint8(circleMask)
            coinList.append(image & circleMask)
            grayCircleMask = cv2.cvtColor(circleMask, cv2.COLOR_BGR2GRAY)
            maskList.append(grayCircleMask)


            if plot:
                # Draw the circumference of the circle.
                cv2.circle(image, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
                cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Detected Circle", 600, 800)
                cv2.imshow("Detected Circle", image)
                cv2.waitKey(0)

        return coinList, maskList
########################################################################################################################
    # Otsu Thresholding
    thresh = threshold_otsu(grayImage)
    binary = np.uint8(grayImage > thresh)
    #edges = filters.prewitt(binary)
    binaryNot = np.uint8(cv2.bitwise_not(binary))
    binaryNot[binaryNot < 255] = 0

    # Contour find & plot
    contours, hierarchy = cv2.findContours(binaryNot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape, np.uint8)
    A = cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    A = np.uint8(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY))

    cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Circle", X, Y)
    cv2.imshow("Detected Circle", A)
    cv2.waitKey(0)


########################################################################################################################

    # Morphological filtering
    print("closing operation in progress...")
    structElement = disk(radius=45)
    closedBinaryImage = np.uint8(ndimage.binary_closing(A, structElement))
    print("closing operation complete.")  # added this verification message because the operation takes a while

    cleared = clear_border(closedBinaryImage)

    # label image regions
    labelImage = label(closedBinaryImage)


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


# function to extract Haralick textures from an image
def extract_features(image, shape, circleMask):
    # # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image, ignore_zeros=True, distance=2)

    # take the mean of each of the four gray-level co-occurrence matrices and return the result
    ht_mean = textures.mean(axis=0)

    if shape:
        area = np.sum(circleMask / 255)
        sobel = np.uint8(255 * filters.edges.sobel(circleMask))
        sobel[sobel > 0] = 255
        perimeter = np.sum(sobel / 255)

    return ht_mean, area, perimeter

    # Use SURF algorithm to calculate texture features
    # spoints = surf.surf(image)
    # s_mean = spoints.mean(axis=0)
    #return s_mean
    #
    # lpoints = mt.features.lbp(image, radius=10, points=6)
    #
    # return lpoints
    # combined_mean = np.concatenate((ht_mean, s_mean))


# function to run through segmented coin list
def Training(ImageList, objectName, trainingFeatures, trainingLabels, reduceFeatures, shape, masks):
    for i in range(len(ImageList)):
        # print("Processing Image - {} for {}".format(i + 1, objectName))
        # read the training image
        image = ImageList[i]

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = masks[i]

        # extract haralick texture from the image
        features, area, perimeter = extract_features(gray, shape, mask)

        if reduceFeatures is not None:
            features = features[:reduceFeatures]

        features = np.insert(features, 0, [area, perimeter])

        trainingFeatures.append(features)

        # append the feature vector and label
        trainingLabels.append(objectName)
    # print('\n')
    return trainingFeatures, trainingLabels


def Testing(inputImage, clf_svm, correctString, plot, reduceFeatures, shape, masks):

    coinPrediction = []
    count = 0
    for a in range(len(inputImage)):
        # read the input image
        image = inputImage[a]
        X, Y, _ = image.shape

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = masks[a]

        # extract haralick texture from the image
        features, area, perimeter = extract_features(gray, shape, mask)
        if reduceFeatures is not None:
            features = features[:reduceFeatures]

        features = np.insert(features, 0, [area, perimeter])

        # # data preprocessing
        # scaler = StandardScaler()
        # test_data = scaler.fit_transform(features)

        # evaluate the model and predict label
        prediction = clf_svm.predict(features.reshape(1, -1))[0]

        # show the label
        # cv2.putText(image, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        # print("Prediction - {}".format(prediction))
        coinPrediction.append(prediction)

        if plot:
            # display the output image
            cv2.namedWindow("Test Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test Image", X, Y)
            cv2.imshow("Test Image", image)
            cv2.waitKey(0)

        correct = correctString[a]
        testOutcome = correct == prediction
        if testOutcome:
            count += 1

    successRate = 100 * (count / len(inputImage))
    print("Success Rate: {} %".format(successRate))

    return coinPrediction, successRate