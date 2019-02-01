## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration/raw.jpg "Raw Image"
[image2]: ./output_images/calibration/undistort.jpg "Undistorted"
[image3]: ./output_images/before_wrap.jpg "Before Wrap"
[image4]: ./output_images/after_wrap.jpg "After Wrap"

[image5]: ./output_images/raw.jpg "Raw Image"
[image6]: ./output_images/undistort.jpg "Undistorted"
[image7]: ./output_images/lane_binary.jpg "Binary Image"
[image8]: ./output_images/lane_binary_wrapped.jpg "Binary Wrapped"
[image9]: ./output_images/lane_line_detection_wrapped.jpg "Lane Fitted"
[image10]: ./output_images/lane_detection_wrapped.jpg "Lane Fitted Area"
[image11]: ./output_images/undistort_with_detected_lane.jpg "Final Result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


The source code for calibaration step is in the lines 8 through 36 of the file called `calibration.py`). The code from lines 40 till the end of file is used for saving purpose.

First step is initiate nx and ny params for chessboard when nx is a corner in the x axis and y is a corner in y axis. 
Second, I prepare `objp` which is coordinates of "object points" in the visual world where z axis is always 0 but x, y is the combination of object point of the chessboard corners and `objp` will be appended into `objpoints` array when `imgpoints` is fully detected on the chessboard. I do that by read all images in the camera_cal folder and use `cv2.findChessboardCorners()` function to detect the chessboard corner and append them into `imgpoints`.
After that, I call `cv2.calibrateCamera()` on the `objpoints` and `imgpoints` data to compute camera calibration matrix and distortion coefficients.
Then, I applied the distortion coefficients and camera calibration matrix into `cv2.undistort()` and obtained the undistort image.

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Firstly, I get distorted image
![alt text][image5]

Then, I appied camera calibration matrix and distortion coefficients using `cv2.undistort` to get.

![alt text][image6]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I combine s channel color in HGS color space with threshold = (170, 255) and x axis sobel operator with threshold = (20, 100). If any color is in the thresold are filled with 1 otherwise 0. The code of the algorithm is in `processing.py` file (lines 97-118) which is `get_road_pixel()` function. The result of this function is in the bottom.

![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform is in the function `wrap()` in the `warper.py` file (lines 8-15). The `src` and `dst` are hardcoded. Which, I adjust the points by hand on the `examples/straight_lines1.jpg`. The `dst` points (200,720), (200,0), (800,0), (800,720) are also picked by hand. The warping result on `examples/straight_lines1.jpg` is below. The `unwrap()` function in the same file is for reversing the wrapped lane area (get from 4) into the front camera. 

![alt text][image3]


![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I detect the lanes by using 2 techniques. 
First, It is used on the first frame. It is a `get_lane_poly_ff()` function in `processing.py` file.
`get_lane_poly_ff()` first get the histogram of bottom half of the wrapped image to get the starting points. Then, I use sliding windows by get all nonzero pixel in the window (lines 29-44) to fit the polynomial in the `polyfit()` function (lines 48-60).

Second, It is used on second frame and after. It is a `get_lane_around_poly()` function in `processing.py` file. It use prior fitted lines as a start points for doing a window searching. 


![The result of polyfit lines][image9]


![After fill area between fit lines][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate radius of curvature in a `get_lane_curvature()` function (lines 73-76) which get lane curvature for each lane line seperatly. Then, I average the radius of each lane (lines 142).

The position of the vehicle is calculated by get x position of the lanes in `get_lane_position_x()` function (lines 78-80) then calculate number of pixel the center of this lanes is difference from center of the image. (lines 141-142)



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the algorithm to plot lane area back by 
1. In the `get_lanes_area()` function. It input a blank image with a same size as camera img. Then, I use `line_pts = np.hstack((left_lane_line, right_lane_line))` to fill the area between lane lines and use `cv2.fillPoly()` to fill lane area into the blank input. 
2. I unwrap persective the filled image back to the raw camera input perspective using `warpper.unwrap()`
3. I use `weighted_img()` in `utils.py` to add raw_img (line 166) `processing.py` and img (line 175) together.
The result is as below.

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video pipeline is in `video.py` file. The pipeline is same as image pipeline which is in `run()` function of `processing.py` file.

Here's a [link to my video result](./out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is mostly to fail on

1. The color and s channel lane lines detection cannot properly detect the lane lines because I didn't implement any algorithm to remove noise or other car when it was detected.
2. The color detection algorithm will classify the suddenly color change on the road as a lane lines because there are gradients there.