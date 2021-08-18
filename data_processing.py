import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


class DataProcessing:

    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None

    def resize(image):
        """ Method that standardizes an image by resizing it to a specific height and weight """
        height = 28
        width = 28
        dimension = (height, width)
        return cv2.resize(image, dimension, interpolation=cv2.INTER_LINEAR)

    def denoise(img):
        """ Function that smooths out the edges of the picture """
        return cv2.GaussianBlur(img, (5, 5), 0)

    def threshold(img, threshold=0.925):
        """
        Clears the background noise from data
        NOTE: Max value after normalization is 1.0
        (0, 0, 0) represents black
        (255, 255, 255) represents white
        Hence, we want to set everything that is not 1.0 to 0.0 to have a clear black background
        To turn background to white and numbers to black simply add _INV to the 4th argument for the threshold function.
        """
        return cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)

    def load_data(self):
        """
        Loads data from .pkl file and prints out a count summary + example of image
        """
        self.train_data = pd.read_pickle('train_max_x')
        self.test_data = pd.read_pickle('test_max_x')
        self.train_label = pd.read_csv('train_max_y.csv')

    def preprocessData(self):
        """
        Preprocesses the modified MNIST images
            - Normalizes RGB values
            - Removes background noise
        """
        # Import data
        self.train_data = pd.read_pickle('train_max_x')
        self.test_data = pd.read_pickle('test_max_x')
        self.train_label = pd.read_csv('train_max_y.csv')

        # Set all values to 0 or 1
        self.train_data, self.test_data = self.train_data / 255.0, self.test_data / 255.0

        # Thresholding both train and test data
        for i in range(len(self.train_data)):
            self.train_data[i] = self.threshold(self.train_data[i])[1]

        for i in range(len(self.test_data)):
            self.test_data[i] = self.threshold(self.test_data[i])[1]

        # Save preprocessed data
        pickle.dump(self.train_data, open(r"train_max_x.pkl_preprocessed_", "wb"))
        pickle.dump(self.test_data, open(r"test_max_x_preprocessed.pkl", "wb"))