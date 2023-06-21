import os
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from preProcessing import roi, savePath

inputPath = savePath
areasOutputPath = 'images/outputs/areas/test'

def applyMorphoOp():
    x, y, w, h = roi
    for filename in os.listdir(inputPath):
        if filename.endswith('.jpg'):
            # read image with only one color space
            img = cv2.imread(os.path.join(inputPath, filename), 0)
        
            # crop image using region of interest
            #img = img[y:y+h, x:x+w]
        
            # apply opening morphological operation to better separate b&w areas
            # using smallest rectangular kernel
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
            # extract contours using simple aproximation - eliminates redundant points and compresses the contours
            contours, hierarchy = cv2.findContours(image=opening, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            img_copy = opening.copy()
            cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.FILLED)

            #cv2.namedWindow('Simple aprox', cv2.WINDOW_NORMAL)
            #width, height = img_copy.shape[:2]
            #cv2.resizeWindow('Simple aprox', width, height)
            #cv2.imshow('Simple aprox', img_copy)
            #cv2.waitKey(0)
        
            # Calculates areas of contours
            areaCalculation(contours, img.shape, filename)
            
            # Extracts dimensions of contours
            length1, length2, color, drawing, rectPoints, resultZero = extractDimensions(contours, img.shape, hierarchy)
            
            for j in range(4):
                if length1 > length2:
                    cv2.putText(img, str(length1), tuple(rectPoints[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                else:
                    cv2.putText(img, str(length2), tuple(rectPoints[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.line(resultZero, tuple(rectPoints[j]), tuple(rectPoints[(j+1) % 4]), color, 1, 8)
            
        else:
            print(filename + 'is not a valid input format.')
    
        
            
def areaCalculation(contourArray, imgShape, fileName):
    blankImg = np.zeros((imgShape), np.uint8)
    imgArea = np.prod(imgShape)
    #fig = plt.figure(constrained_layout=True)
    #fig.suptitle('Isolated contours')
    
    # set number of columns (use 3 to demonstrate the change)
    #ncols = 3
    # calculate number of rows
    #nrows = len(contourArray) // ncols + (len(contourArray) % ncols > 0)
    
    
    for i, c in enumerate(contourArray[::-1]):
        # turn blank_image black
        blankImg *= 0

        # draw filled contour
        cv2.drawContours(image=blankImg, contours=[c], contourIdx=0, color=(255), thickness=cv2.FILLED)
        contourArea = cv2.contourArea(c)
        
        # percentage of area contour
        if int(contourArea) > 1:
            contourAreaPerc = np.true_divide(int(contourArea), imgArea)*100
        else:
            contourAreaPerc = 0

        if contourAreaPerc > 0.2:
            text = ' '.join(['Contour:', str(i), 'Area:', str(round(contourArea, 2)), '\n', 'Percentage Area:', str(round(contourAreaPerc, 2))])
            ret, thContour = cv2.threshold(blankImg, 0, 255, cv2.THRESH_BINARY)  #remove for prev state
            plt.imshow(thContour, cmap='gray', interpolation='bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.suptitle(text)
            plt.axis('off')
            #plt.show()
            #ax = fig.add_subplot(nrows, ncols, i + 1)
            #ax.imshow(blankImg)
            #ax.axis('off')
            #ax.set_title(text)
            baseName, extension = os.path.splitext(fileName)
            newFileName = baseName + '_contour_' + str(i) + extension
            plt.savefig(os.path.join(areasOutputPath, newFileName))
        else:
            print('no significant area in contour ', i)
    #plt.show()
            
def extractDimensions(contourArray, imgShape, hierarchy):
    rng = np.random.RandomState(12345)
    minRect = [cv2.minAreaRect(contour) for contour in contourArray]
    drawing = np.zeros(imgShape, dtype=np.uint8)
    resultZero = np.zeros(imgShape, dtype=np.uint8)
    
    for i, contour in enumerate(contourArray):
        color = tuple(rng.uniform(0, 255, size=3).astype(int).tolist())
        
        # Detect contours
        cv2.drawContours(drawing, [contour], 0, color, 1, 8, hierarchy, 0)
        
        # Detect rectangle for each contour
        rectPoints = cv2.boxPoints(minRect[i]).astype(int)
        
        length1 = np.linalg.norm(rectPoints[0] - rectPoints[1])
        length2 = np.linalg.norm(rectPoints[1] - rectPoints[2])
        
        print('length 1: ', length1)
        print('length 2: ', length2)
        
        return length1, length2, color, drawing, rectPoints, resultZero
                    
applyMorphoOp()