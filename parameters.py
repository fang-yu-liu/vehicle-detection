COLOR_SPACE = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 12  # HOG orientations
PIX_PER_CELL = 8 # HOG pixels per cell
CELL_PER_BLOCK = 2 # HOG cells per block
HOG_CHANNEL = 'ALL' # Can be 0, 1, 2, or "ALL"
SPATIAL_SIZE = (32, 32) # Spatial binning dimensions
HIST_BINS = 32    # Number of histogram bins
SPATIAL_FEAT = True # Spatial features on or off
HIST_FEAT = True # Histogram features on or off
HOG_FEAT = True # HOG features on or off

X_START_STOP = [200, None]
Y_START_STOP = [380, 600] # Min and max in y to search in slide_window()
XY_WINDOW = (64, 64)
XY_OVERLAP = (0.5, 0.5)
HEAT_THRESH = 30

SAVE_MODEL= True
MODEL_FILE = "./model.p"

RETRAIN_MODEL = True