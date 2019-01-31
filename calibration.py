import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

nx = 9
ny = 6


objpoints = []
imgpoints = []

objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

images = glob.glob('camera_cal/*.jpg')
gray = None

for fname in images:
    print(fname)
    img = mpimg.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # add detected point to array
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)


# calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# dump data
pickle.dump(mtx, open( 'pickle/camera_param.p', 'wb' ) )
pickle.dump(dist, open('pickle/distort_param.p', 'wb') )

test_img = mpimg.imread('camera_cal/calibration1.jpg')

# undistort and show
test_undist = cv2.undistort(test_img, mtx, dist)
plt.imsave('output_images/calibration/raw.jpg', test_img)
plt.imsave('output_images/calibration/undistort.jpg', test_undist)
plt.imshow(test_undist)
plt.show()

