COLOR_SPACE = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 9  # HOG orientations
PIX_PER_CELL = 8 # HOG pixels per cell
CELL_PER_BLOCK = 2 # HOG cells per block
HOG_CHANNEL = 0 # Can be 0, 1, 2, or "ALL"
SPATIAL_SIZE = (16, 16) # Spatial binning dimensions
HIST_BINS = 16    # Number of histogram bins
SPATIAL_FEAT = True # Spatial features on or off
HIST_FEAT = True # Histogram features on or off
HOG_FEAT = True # HOG features on or off

X_START_STOP = [None, None]
Y_START_STOP = [350, None] # Min and max in y to search in slide_window()
XY_WINDOW = (96, 96)
XY_OVERLAP = (0.5, 0.5)

SAVE_MODEL= True
MODEL_FILE = "./model.p"

RETRAIN_MODEL = True