from skimage.feature import hog
import cv2
import numpy as np

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    '''
    Extract HOG features and visualization.
    
    Input:
        img: The input image.
        orient: The number of orientation bins.
        pix_per_cell: The size (in pixels) of a cell.
        cell_per_block: The number of cells in each block.
        vis: Boolean, determine whether the function should return an image of the HOG.
        feature_vec: Boolean, determine whether the function should return the data as a feature vector by calling .ravel() on the result just before returning.
    Output:
        features: HOG descriptor for the image. If feature_vector is True, a 1D (flattened) array is returned.
        hog_image (optional): A visualisation of the HOG image. Only returned if vis is True.
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def color_conversion(image, color_space):
    '''
    Perform color space conversion.
    
    Input:
        image: input image in 'RGB' color space
        color_space: the color space for the output image
    Output:
        color_converted: the image in the given color space
    '''
    # apply color conversion if other than 'BGR'
    if color_space != 'RGB':
        if color_space == 'HSV':
            color_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            color_converted = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            color_converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            color_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            color_converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: 
        color_converted = np.copy(image)  
    return color_converted

def bin_spatial(img, size=(32, 32)):
    '''
    Perform spatial binning.
    
    Input:
      img: Tnput image.
      size: The resolution to resize to.
    
    Output:
      features: The 1D vector to represent the features after down scaling.
    '''
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Compute color histogram features.
    
    Input:
      img: The input image.
      nbins: The number of the bins.
      bins_range: The lower and upper range of the bins.
    
    Output:
      hist_features: The color histogram features.
    
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    # channel1_hist[0]: the counts in each of the bins / channel1_hist[1]: the bin edges 
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range) 
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
   
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def single_img_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Perform feature extraction.
    
    Input:
        image: input image in 'RGB' color space
        color_space: the color space for the output image
        spatial_size: see bin_spatial
        hist_bins: see color_hist
        orient: see get_hog_features
        pix_per_cell: see get_hog_features
        cell_per_block: see get_hog_features
        hog_channel: see get_hog_features
        spatial_feat: whether to perform spatial spinning
        hist_feat: whether to perfom color histogram extraction
        hog_feat: whether to perform hog feature extraction
    Output:
        features: the extracted features
    '''
    img_features = []
    # apply color conversion
    feature_image = color_conversion(image, color_space=color_space)     

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        img_features.append(hog_features)
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(filenames, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9, 
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Perform feature extraction on a list of image filenames.
    
    Input:
        image: images filenames
        color_space: see single_img_features
        spatial_size: see single_img_features
        hist_bins: see single_img_features
        orient: see single_img_features
        pix_per_cell: see single_img_features
        cell_per_block: see single_img_features
        hog_channel: see single_img_features
        spatial_feat: see single_img_features
        hist_feat: see single_img_features
        hog_feat: see single_img_features
    Output:
        features: the extracted features
    '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for filename in filenames:
        # Read in each one by one using cv2.imread() -> the decoded color channel will be in BGR order
        image = cv2.imread(filename) 
        # Conver to RGB color space
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        file_features = single_img_features(rgb_image, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        features.append(file_features)
    # Return list of feature vectors
    return features