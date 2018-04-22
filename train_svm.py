from features import extract_features
from parameters import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time
import pickle

svc = LinearSVC()
X_scaler = StandardScaler()

def train(car_filenames, notcar_filenames):
    car_features = extract_features(car_filenames, 
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
    notcar_features = extract_features(notcar_filenames, 
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

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler.fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:',ORIENT,'orientations',PIX_PER_CELL,
        'pixels per cell and', CELL_PER_BLOCK,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
            
if __name__ == '__main__':
    if RETRAIN_MODEL is True:
        # Use the following two lines to generate txt files to store the filenames for vehicle and non-vehicle data
        # find vehicles -name '*.png' > vehicle-images.txt
        # find non-vehicles -name '*.png' > non-vehicle-images.txt
        # Iterate through the txt file and save the filenames as a list
        with open('./vehicle-images.txt', 'r') as f:
            car_filenames = f.read().splitlines() 
        n_cars = len(car_filenames)
        print("Numbers of vehicle dataset:", n_cars)

        with open('./non-vehicle-images.txt', 'r') as f:
            notcar_filenames = f.read().splitlines()
        n_notcars = len(notcar_filenames)
        print("Numbers of non vehicle dataset:", n_notcars)

        print("===== Start training model =====")
        train(car_filenames, notcar_filenames)
        print("===== End training model =====")
        
        if SAVE_MODEL is True:
            model_data = {'svc': svc, 'X_scaler': X_scaler}
            with open(MODEL_FILE, mode='wb') as f:
                pickle.dump(model_data, f)
            print("Saved trained model to", MODEL_FILE)
