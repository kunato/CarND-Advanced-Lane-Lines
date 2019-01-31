import cv2

def weighted_img(img, initial_img, alpha=0.8, beta=0.3, gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)