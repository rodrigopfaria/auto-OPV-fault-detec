import numpy as np
import cv2 as cv
import glob

# Define the dimensions of checkerboard
CHECKERBOARD = (9, 7)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)

prev_img_shape = None

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = ['images/calibration/*.jpg']

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if (ret is True):
        objpoints.append(objp)
        # Refining pixel coordinates
        # for given 2d points.
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)

cv.destroyAllWindows()

# Perform camera calibration by
# passing the value of above found out 3D points (objpoints)
# and its corresponding pixel coordinates of the
# detected corners (imgpoints)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,
                                                  imgpoints,
                                                  gray.shape[::-1],
                                                  None, None)

print("Camera matrix: ")
print(mtx)

print("\n Distortion coefficient:")
print(dist)

print("\n Rotation Vectors:")
print(rvecs)

print("\n Translation Vectors:")
print(tvecs)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i],
                                     tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(objpoints)))

# undistorting image
img = cv.imread('images/calibration/close2.jpg')
h, w = img.shape[:2]
# cv.imshow('img', img)
# cv.waitKey(0)
# refining camera matrix
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i],
                                     tvecs[i], newcameramtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error with new matrix: {}".format(mean_error/len(objpoints)))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# cv.imshow('calib', dst)
# cv.imwrite('images/outputs/calibresult.jpg', dst)

print("New camera matrix:")
print(newcameramtx)

print('roi:')
print(roi)