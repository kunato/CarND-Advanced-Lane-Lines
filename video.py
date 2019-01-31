from moviepy.editor import VideoFileClip
from processing import Road, Lane
import numpy as np
import utils

class VideoProcessing:
    def __init__(self):
        self.road = Road(prior=True)

    def process_image(self, img):
        img = self.road.get_undistort(img)
        raw_img = np.copy(img)
        blank_img = np.zeros_like(img)
        img = self.road.get_road_pixel(img)
        img = self.road.get_road_wrapped(img)
        lane_img = self.road.get_lanes_area(img, blank_img)
        lane_img = self.road.get_road_unwrapped(lane_img)
        out_img = utils.weighted_img(lane_img, raw_img)
        out_img = self.road.put_vehicle_stat(out_img)
        return out_img

clip1 = VideoFileClip('project_video.mp4')
video = VideoProcessing()
white_clip = clip1.fl_image(video.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile('out.mp4', audio=False)