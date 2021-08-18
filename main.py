from data_processing import DataProcessing
from model_selection import ModelSelection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle


def main():

    """
    DATA PREPROCESSING
    """
    # Preprocesses data and saves to new files
    dp = DataProcessing()
    dp.preprocessData()

    # Reads and stores preprocessed images
    x = pd.read_pickle('train_max_x_preprocessed.pkl')
    y = pd.read_csv("train_max_y.csv")['Label']

    # Splits data for validation
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True)

    # Saves as separate files
    pickle.dump(x_train, open(r"x_train.pkl", "wb"), protocol=4)
    pickle.dump(x_valid, open(r"x_valid.pkl", "wb"), protocol=4)
    pickle.dump(y_train, open(r"y_train.pkl", "wb"), protocol=4)
    pickle.dump(y_valid, open(r"y_valid.pkl", "wb"), protocol=4)

    # Convert training/valid images from grayscale to RGB (3 channels)
    x_train = np.stack((x_train, x_train, x_train), axis=-1)
    x_valid = np.stack((x_valid, x_valid, x_valid), axis=-1)

    # Saves image dimensions
    image_size = x_train[0].shape[0]
    num_channels = x_train[0].shape[2]

    """
    MODEL CREATION
    (fit, predict)
    """
    # Creates model
    model = ModelSelection()
    model.fit_image_generator(x_train)
    model.init_model()

    history = model.fit_image_generator(x_train, y_train)

    # Read and reshape test data
    x_test = pickle.load(open('test_max_x_preprocessed.pkl', 'rb'))
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    model.predict(x_test)


    """
    MODEL ANALYSIS 
    (graphing)
    """
    # Plot loss
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('CNN_loss.png')
    plt.show()

    # Plot accuracy
    plt.plot(history.history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('CNN_accuracy.png')
    plt.show()

    # Save the model
    model.save('CNN.h5')

if __name__ == '__main__':
    main()