import numpy as np
import pickle
from windows import slide_window, search_windows
from annotate import draw_boxes, draw_labeled_bboxes
from heat import add_heat, apply_threshold
from scipy.ndimage.measurements import label
from parameters import *

def process(img):
    image = np.copy(img)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    windows = slide_window(image, x_start_stop=X_START_STOP, y_start_stop=Y_START_STOP, 
                           xy_window=XY_WINDOW, xy_overlap=XY_OVERLAP)

    with open(MODEL_FILE, mode='rb') as f:
        model_data = pickle.load(f)
    model = model_data['svc']
    X_scaler = model_data['X_scaler']
    
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
    
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    annotated_image = draw_labeled_bboxes(image, labels, color=(0, 0, 255), thick=6) 
    return annotated_image

