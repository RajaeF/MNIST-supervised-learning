import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split

# Using pre-trained models
from keras.applications import VGG16, VGG19, InceptionResNetV2, Xception, NASNetLarge
from keras import datasets, optimizers, Sequential
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


class ModelSelection:

    def __init__(self):
        self.model = None
        self.datagen = None

        self.epochs = 30
        self.batch_size = 256
        self.learning_rate = 0.0005
        self.optimizer = optimizers.SGD(lr=self.learning_rate, momentum=0.8)


    def createImageGen(self, X_train):
        # Create datagen to improve RAM use
        self.datagen = ImageDataGenerator(rotation_range=0, zoom_range=0.10,
                                     width_shift_range=0.05, height_shift_range=0.05, shear_range=0,
                                     horizontal_flip=False, vertical_flip=False, fill_mode="nearest")

        # Fit data generator on training data
        self.datagen.fit(X_train)

    def fitImageGen(self, X_train, y_train):
        steps_per_epoch = len(X_train) / self.batch_size
        history = self.model.fit_generator(self.datagen.flow(x=X_train, y=y_train, batch_size=self.batch_size),
                                      steps_per_epoch=steps_per_epoch, epochs=self.epochs)
        return history


    def init_model(self, image_size, num_channels):
        """ Function that returns a model """
        self.model = Sequential()
        self.model.add(Conv2D(128, (3, 3), input_shape=(image_size, image_size, num_channels), activation='relu',
                         data_format="channels_last"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', data_format="channels_last"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.20))
        self.model.add(Conv2D(64, (3, 3), activation='relu', data_format="channels_last"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.20))
        self.model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_last"))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dropout(0.20))
        self.model.add(Dense(10, activation='softmax'))

        # Print model data
        self.model.summary()

        # Compile model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return self.model

    def predict(model):
        """ Method that takes as input a pretrained model """
        # Read test data
        x = pickle.load(open('test_max_x_preprocessed.pkl', 'rb'))


        # Reshape test
        x = x.reshape(x.shape[0],128,128)
        x = np.stack((x,x,x), axis=-1)


        # Predict
        predictions = model.predict(x)
        predictions = np.argmax(predictions,1)


        # Save predictions
        df = pd.DataFrame({"Id": list(range(len(predictions))), "Label": predictions})
        df.to_csv("predict.csv", index=False)

    def plot_accuracy(self, history):
        plt.figure(figsize=(15, 15))
        plt.plot(history.history['acc'], label='accuracy')
        plt.plot(history.history['val_acc'], label='val_accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0.5, 1])
        plt.legend(loc=0)
        plt.savefig("plots/train_val_accuracy{}.png".format(datetime.datetime.now()))
        plt.show()
