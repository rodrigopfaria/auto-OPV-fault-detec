# pylint: disable=trailing-whitespace
# pylint: disable=trailing-whitespace
# pylint: disable=invalid-name

""" This component is to be used at the start of the OPV cell area calculation 
    to inport the images and transform them into usable formats"""

import os 
import cv2
import numpy as np
#from cameraCalibration import camMatrix, distCoefs, newCamMatrix, roi

inputPath = 'images/inputs/raw/ok/'
savePath = 'images/inputs/std_cam_mtx/'
thresholdValue = 180
camMatrix = np.array ([[16179.8308, 0, 1338.0936], [0, 15052.138, 862.633965], [0, 0, 1]])
newCamMatrix = np.array([[16010.2676, 0, 1333.56955], [0, 14860.2578, 862.684898], [0, 0, 1]])
distCoefs = np.array([[0.211915753, -276.324607, -0.0126142388, -0.064019359, 1.78382264]])
roi = (16, 13, 1898, 1064)

# add threshold value, calibration data, and file path as parameters later
def preProcess():
    for filename in os.listdir(inputPath):
        if filename.endswith('.jpg'):
            # Read the image and convert it to grayscale
            image = cv2.imread(os.path.join(inputPath, filename))
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            ret, thresholded = cv2.threshold(grayscale, thresholdValue, 255, cv2.THRESH_BINARY)
        
            # Undistort the image
            undistorted = cv2.undistort(thresholded, camMatrix, distCoefs, None)
            
            # Save the grayscale image to the new folder
            baseName, extension = os.path.splitext(filename)
            newFileName = baseName + '_th' + extension
            saveFileName = os.path.join(savePath, newFileName)
            cv2.imwrite(saveFileName, undistorted)
        else:
            print(filename + 'is not a valid input format.')
            

preProcess()