import cv2
import numpy as np

from skimage import filters
from skimage import io
import matplotlib.pyplot as plt
from skimage.draw import circle
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
import mahotas as mt

def CoinSegmentation(imageName, Hough):
    image = cv2.imread(imageName)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.medianBlur(grayImage, 25)
    ########################################################################################################################

    if Hough:
        # Testing Circular Hough Transform
        print("performing circular Hough transform...")
        circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 120, param1=60, param2=30, minRadius=20, maxRadius=45)
        if circles is None:
            return None
        circles = np.uint16(np.around(circles))
        print("circular Hough transform complete...")

        coinList = []
        for pt in circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            rr, cc = circle(a, b, r)
            circleMask = np.zeros(image.shape)
            circleMask[cc, rr] = 255
            circleMask = np.uint8(circleMask)
            coinList.append(image & circleMask)

            # Draw the circumference of the circle.
            # cv2.circle(image, (a, b), r, (0, 255, 0), 2)
            #
            # # Draw a small circle (of radius 1) to show the center.
            # cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
            # cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("Detected Circle", 600, 800)
            # cv2.imshow("Detected Circle", image)
            # cv2.waitKey(0)
        return coinList
########################################################################################################################
    # Otsu Thresholding
    thresh = threshold_otsu(grayImage)
    binary = np.uint8(grayImage > thresh)
    #edges = filters.prewitt(binary)
    binaryNot = np.uint8(cv2.bitwise_not(binary))

    # io.imshow(binaryNot)
    # plt.show()

########################################################################################################################

    # Morphological filtering
    structElement = disk(radius=30)
    closedBinaryImage = np.uint8(closing(binaryNot, structElement))
    print("closing operation complete.")  # added this verification message because the operation takes a while

    cleared = clear_border(closedBinaryImage)

    # label image regions
    # labelImage = label(closedBinaryImage)
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


# function to extract Haralick textures from an image
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of each of the four gray-level co-occurrence matrices and return the result
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