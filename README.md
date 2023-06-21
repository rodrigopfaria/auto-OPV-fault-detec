# auto-OPV-fault-detec
Algorithm based on PythonCV developed for my University Final Project that detects faults in OPV films.
Only one input image is included, the rest of the database was removed due to the confidentiality of the material and the NDA between the company for which I developed this test and their client.


# How to use:
The first step is to set up the following folder structure:
- images/
-   calibration/
-   inputs/
-     bw_threshold/
-     raw/
-   outputs/
-     areas/
-     calibration/
-     contours/
-     thresholds/
  
Save the input images for analysis in the "images/inputs/raw/" subfolder in .jpg format.
Install the needed libraries in your virtual or global environment: numpy, cv2, pandas, matplotlib, glob.

1) Camera calibration
  If you are using an ELP-USBFHD08S-MFV camera, running the calibration script or even doing the calibration protocol is unnecessary. Skip to step 2.
  For other cameras:
    - set up a calibration checkerboard pattern (use an online generator) on a fixed point and capture several images by moving the camera around. The pattern should be 9x7.
    - save the calibration input images in the "images/calibration/" subfolder in .jpg format.
    - run cameraCalibration.py

2) Adjust threshold
   Modify the contourTest.py scrip to use one or all of the raw input images and adjust the threshold value in line 7 (second parameter) until the regions in the OPV film are clearly distinguishable

3) Pre-processing
   You can either import the calibration parameters (remove the comment from line 11) or copy the outputs from that scrip to the variables in lines 16-19.
   Run preProcessing.py

4) Area Extraction
   Run areaExtraction.py
   Results will be saved in the "images/output/areas" subfolder
   
