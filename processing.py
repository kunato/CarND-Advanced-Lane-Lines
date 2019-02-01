import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import warper
import utils
import pickle

class Lane:
    def __init__(self, img, is_left):
        self._img = img
        self.img_height = img.shape[0]
        self.img_width = img.shape[1]
        self.num_window = 9
        self.midpoint = np.int(self.img_height//2)
        self.window_height = np.int(self.img_height//self.num_window)
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        self.is_left = is_left
        self.ym_per_pix = 30.0/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    def get_lane_poly_ff(self):
        histogram = np.sum(self._img[self.img_height//2:,:], axis=0)
        base = np.argmax(histogram[:self.midpoint]) if self.is_left else np.argmax(histogram[self.midpoint:]) + self.midpoint
        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = self._img.nonzero()
        current = base
        lane_inds = []
        for i in range(self.num_window):
            window_x = (current - self.margin, current + self.margin)
            bottom_window_y = self.img_height - (i * self.window_height)
            top_window_y = self.img_height - ((i + 1) * self.window_height)

            current_inds = ((nonzero[0] >= top_window_y) & (nonzero[0] < bottom_window_y) & (nonzero[1] >= window_x[0]) &  (nonzero[1] < window_x[1])).nonzero()[0]
            lane_inds.append(current_inds)

            if len(current_inds) > self.minpix:
                current = np.int(np.mean(nonzero[1][current_inds]))
            

        lane_inds = np.concatenate(lane_inds)

        self.fit, _ , _  = self.polyfit(nonzero, lane_inds)

    def polyfit(self, nonzero, lane_inds, is_meters=False):
        x = nonzero[1][lane_inds]
        y = nonzero[0][lane_inds]
        fit = np.polyfit(y, x, 2) if not is_meters else np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        ploty = np.linspace(0, self._img.shape[0]-1, self._img.shape[0])
        fitx = None
        try:
            fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            fitx = 1 * ploty ** 2 + 1 * ploty
        return fit, fitx, ploty
    
    
    def get_lane_around_poly(self, img, flip=False):
        nonzero = img.nonzero()
        lane_inds = ((nonzero[1] > (self.fit[0] * (nonzero[0] ** 2) + self.fit[1] * nonzero[0] + 
                self.fit[2] - self.margin)) & (nonzero[1] < (self.fit[0]*(nonzero[0]**2) + 
                self.fit[1] * nonzero[0] + self.fit[2] + self.margin)))
        
        self.fit, self.fitx, self.ploty = self.polyfit(nonzero, lane_inds)
        self.fit_meters, _, _ = self.polyfit(nonzero, lane_inds, is_meters=True)
        return np.array([np.flipud(np.transpose(np.vstack([self.fitx, self.ploty])))]) if flip else np.array([np.transpose(np.vstack([self.fitx, self.ploty]))])

    def get_lane_curvature(self):
        y_eval = np.max(self.ploty)
        curverad = ((1 + (2 * self.fit_meters[0] * y_eval * self.ym_per_pix + self.fit_meters[1]) ** 2) ** 1.5) / np.absolute(2 * self.fit_meters[0])
        return curverad

    def get_lane_position_x(self):
        y_eval = np.max(self.ploty)
        return self.fitx[int(y_eval)]

    

class Road:
    def __init__(self, prior=False):
        self.use_prior = prior
        self.is_ff = True
        self.left_lane = None
        self.right_lane = None
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    def get_undistort(self, img):
        camera_mtx = pickle.load(open('pickle/camera_param.p', 'rb'))
        camera_distort = pickle.load(open('pickle/distort_param.p', 'rb'))
        return cv2.undistort(img, camera_mtx, camera_distort)

    def get_road_pixel(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        thresh_sobel = (20, 100)
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= thresh_sobel[0]) & (scaled_sobel <= thresh_sobel[1])] = 1

        thresh_s = (170, 255)
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > thresh_s[0]) & (s_channel <= thresh_s[1])] = 1

        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
        # color_binary = np.dstack(( sobel_binary, np.zeros_like(sobel_binary), s_binary)) * 255
        # plt.imsave('output_images/temp.jpg', color_binary)
        return combined_binary

    def get_road_wrapped(self, img):
        return warper.warp(img)

    def get_road_unwrapped(self, img):
        return warper.unwarp(img)


    def get_lanes_area(self, img, out_img):
        if self.is_ff or not self.use_prior:
            self.left_lane = Lane(img, True)
            self.right_lane = Lane(img, False)
            self.left_lane.get_lane_poly_ff()
            self.right_lane.get_lane_poly_ff()
            self.is_ff = False

        left_lane_line = self.left_lane.get_lane_around_poly(img)
        left_curverad = self.left_lane.get_lane_curvature()
        right_lane_line = self.right_lane.get_lane_around_poly(img,flip=True)
        right_curverad = self.right_lane.get_lane_curvature()
        left_position_x = self.left_lane.get_lane_position_x()
        right_position_x = self.right_lane.get_lane_position_x()
        center_x = right_position_x - left_position_x
        diff_from_center_pix = img.shape[1]/2 - center_x
        self.diff_from_center_m = diff_from_center_pix * self.xm_per_pix
        self.avg_curverad = (left_curverad + right_curverad) / 2

        line_pts = np.hstack((left_lane_line, right_lane_line))
        cv2.fillPoly(out_img, np.int_([line_pts]), (0,255, 0))
        return out_img

    def put_vehicle_stat(self, out_img):
        if self.diff_from_center_m > 0:
            cv2.putText(out_img,'Vehicle is ' + str(self.diff_from_center_m) + '(m) right of center' ,(50,50),  cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        else:
            cv2.putText(out_img, 'Vehicle is ' + str(np.absolute(self.diff_from_center_m)) + '(m) left of center' ,(50,50),  cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        cv2.putText(out_img, 'Radius of Curvature is '+ str(self.avg_curverad) +'(m)', (50,100),  cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
        return out_img


def run():
    # Script for image pipeline
    img = mpimg.imread('test_images/test6.jpg')
    plt.imsave('output_images/raw.jpg', img)

    road = Road()
    img = road.get_undistort(img)
    raw_img = np.copy(img)
    blank_img = np.zeros_like(img)

    plt.imsave('output_images/undistort.jpg', img)
    img = road.get_road_pixel(img)
    plt.imsave('output_images/lane_binary.jpg', img)
    img = road.get_road_wrapped(img)
    plt.imsave('output_images/lane_binary_wrapped.jpg', img)
    img = road.get_lanes_area(img, blank_img)
    img = road.get_road_unwrapped(img)
    out_img = utils.weighted_img(img, raw_img)
    out_img = road.put_vehicle_stat(out_img)
    plt.imsave('output_images/undistort_with_detected_lane.jpg',out_img)
    plt.imshow(out_img)
    plt.show()

# run()