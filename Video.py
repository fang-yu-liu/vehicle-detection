import numpy as np
import pickle
import cv2
from windows import slide_window, search_windows
from annotate import draw_boxes, draw_labeled_bboxes
from heat import add_heat, apply_threshold
from scipy.ndimage.measurements import label
from parameters import *

class Video():
    
    def __init__(self, num_frames):
        # number of frames to calculate the heat map
        self.n = num_frames
        # windows over the last n frames
        self.windows = []
    
    def add_to_windows(self, hot_windows):
        self.windows.append(hot_windows)
        if len(self.windows) > self.n:
            self.windows.pop(0)
            
    def get_windows(self):
        all_windows = []
        for window in self.windows:
            for sub_window in window:
                all_windows.append(sub_window)
        return all_windows
    
    def process(self, img):
        image = np.copy(img)
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        
        with open(MODEL_FILE, mode='rb') as f:
            model_data = pickle.load(f)
        model = model_data['svc']
        X_scaler = model_data['X_scaler']
        
        windows = slide_window(image, x_start_stop=X_START_STOP, y_start_stop=Y_START_STOP, 
                               xy_window=XY_WINDOW, xy_overlap=XY_OVERLAP)

        hot_windows = search_windows(image, windows, model, X_scaler, 
                                     color_space=COLOR_SPACE, 
                                     spatial_size=SPATIAL_SIZE, 
                                     hist_bins=HIST_BINS, 
                                     orient=ORIENT, 
                                     pix_per_cell=PIX_PER_CELL, 
                                     cell_per_block=CELL_PER_BLOCK, 
                                     hog_channel=HOG_CHANNEL, 
                                     spatial_feat=SPATIAL_FEAT, 
                                     hist_feat=HIST_FEAT, 
                                     hog_feat=HOG_FEAT)

        self.add_to_windows(hot_windows)
        # Add heat to each box in box list
        heat = add_heat(heat, self.get_windows())
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, HEAT_THRESH)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        annotated_image = draw_labeled_bboxes(image, labels, color=(0, 0, 255), thick=6) 
        return annotated_image

