# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Detect vehicles using Histogram of Oriented Gradients (HOG) features and Support Vector Machine (SVM) Classifier with sliding window technique.
This project is part of the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/drive) program, and some of the code are leveraged from the lecture materials.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicle]: ./output_images/vehicle.png
[non-vehicle]: ./output_images/non-vehicle.png
[hog]: ./output_images/hog_feature_YUV.png
[window_search]: ./output_images/window_search_vis.png
[heatmap]: ./output_images/heat_map_vis.png
[label]: ./output_images/label_vis.png
[labeled_boxes]: ./output_images/labeled_boxes_vis.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I downloaded the Udacity provided [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) dataset and run the following two commands `find vehicles -name '*.png' > vehicle-images.txt`, `find non-vehicles -name '*.png' > non-vehicle-images.txt` to save the filename for vehicle and non-vehicle images to the txt files. Then the two txt files can be used to read in all the images.

Example of vehicle image | Example of non-vehicle image
------------------ | -----------------
![vehicle][vehicle] | ![non-vehicle][non-vehicle]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

The code for the color conversion is contained in line 4o through 64 of `features.py` and the code for getting HOG features is in line 5 through 38.

Here is an example using the `YUV` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

First, I tried to visualize the hog features using the following 5 different color spaces ('HSV', 'LUV', 'HLS', 'YUV', 'YCrCb') on random images from the dataset. (See code cell 2 and 3 in `hog_features.ipynb`). I first chose the `YCrCb` since it seems to be well capturing the car features from the visualization. And I implemented the pipeline and train the model using `YCrCb`, the result was quite good, it reached 0.98 accuracy score for the test set. I was then curious about the `YUV` color space since from the visualization, it looks quite similar to `YCrCb`. It turns out it's better than `YCrCb` since it reached an accuracy near 0.99. So I decided to use `YUV` for my final implementation.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I chose the linear SVM as my classifier. The code for training the model is in `train_svm.py`.
In the `train()` function, I first extract the features from all the vehicle and non-vehicle images. The code for extracting the features is in `extract_features()` function in `features.py` (line 157 through 195).

Feature extraction:
1. Color conversion (COLOR_SPACE = 'YUV')
2. Spatial binning (SPATIAL_SIZE = (32, 32))
3. Color histogram (HIST_BINS = 32)
4. Hog features (ORIENT = 12, PIX_PER_CELL = 8, CELL_PER_BLOCK = 2, HOG_CHANNEL = 'ALL')

The parameters for those feature extraction are defined in `parameters.py`.

I then split the features data into training set and testing set. And I normalize the features using scikit-learn's `StandardScaler`.

The normalized data was feeded into the scikit-learn's `LinearSVC` to train the model. The model result was saved to `model.p` for future use. Final test accuracy for the trained model was: 98.87%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the basic sliding window technique to search the cars in the image using the following parameters: (See `windows.py` and line 39 through 52 in `Video.py`)

* X_START_STOP = [200, None]
* Y_START_STOP = [380, 600]
* XY_WINDOW = (64, 64)
* XY_OVERLAP = (0.5, 0.5)

Parameters for the sliding window search are also defined in `parameters.py`.

Since in the video, my car is always on the left lane, I decided to start the search at x = 200 to avoid detecting the cars from the opposite lane. Also starting the search at y = 380 to avoid the noise from trees and sky.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on only one scale using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are the window search output for the test images:

![window_search][window_search]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![heatmap][heatmap]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![label][label]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![labeled_boxes][labeled_boxes]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The issue I faced in my implementation is that sometimes the box only covers part of the car instead of the whole car. The following are the potential ways to improve this issue:
* Fine-tune feature extraction parameters
* Increase overlapping area for sliding window
* Decrease heat thresh (but might cause false positive result to show up)
* Implement multi-scale window search
