import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pickle


def warp(img):
    src = np.array([[210,720],[1100,720],[593,450],[688,450]], dtype=np.float32)
    dst = np.array([[200,720],[800,720],[200,0],[800,0]], dtype=np.float32)
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def unwarp(img):
    src = np.array([[210,720],[1100,720],[593,450],[688,450]], dtype=np.float32)
    dst = np.array([[200,720],[800,720],[200,0],[800,0]], dtype=np.float32)
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped


def run():
    img = mpimg.imread('test_images/straight_lines1.jpg')
    camera_mtx = pickle.load(open('pickle/camera_param.p','rb'))
    camera_distort = pickle.load(open('pickle/distort_param.p','rb'))
    img = cv2.undistort(img, camera_mtx, camera_distort)
    cv2.line(img,(210,720),(593,450),(255,0,0),3)
    cv2.line(img,(1100,720),(688,450),(255,0,0),3)
    mpimg.imsave('output_images/before_wrap.jpg', img)
    img2 = warp(img)

    cv2.line(img2,(200,720),(200,0),(255,0,0),3)
    cv2.line(img2,(800,720),(800,0),(255,0,0),3)
    mpimg.imsave('output_images/after_wrap.jpg', img2)

# run()