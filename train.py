#!/usr/bin/env python
# coding: utf-8
import time
import keras
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint

imageSize = 28
classes = 66
channels = 1
epochs = 100
path = 'symbols'


def build_model():
    model = Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(classes, activation=tf.nn.softmax))
    
    return model

def mnist_cnn_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(imageSize, imageSize, channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_data():
    trainDataset = keras.preprocessing.image_dataset_from_directory(
        directory=path,
        image_size=(28, 28),
        labels='inferred',
        label_mode='int',
        class_names=list(map(lambda x: str(x), [*range(0, classes)])),
        color_mode='grayscale',
        batch_size=64,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='training',
        interpolation='bilinear',
        follow_links=False,
    )
    
    valDataset = keras.preprocessing.image_dataset_from_directory(
        directory=path,
        image_size=(imageSize, imageSize),
        labels='inferred',
        label_mode="int",
        class_names=list(map(lambda x: str(x), [*range(0, classes)])),
        color_mode='grayscale',
        batch_size=64,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset='validation',
        interpolation='bilinear',
        follow_links=False,
    )
    
    def normalize(image, label):
        image = tf.cast(image/255., tf.float32)
        return image, label

    trainDataset = trainDataset.map(normalize)
    valDataset = valDataset.map(normalize)
    
    return trainDataset, valDataset


def train(model):
    trainData, valData = generate_data()
    model.compile(optimizer=Adam(lr=0.005), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    checkpoint = ModelCheckpoint('models/model_' + time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime()) + '.h5',
                                 monitor='val_acc', verbose=1,
                                 save_only_best=True, mode='max', period=2)
    model.fit(trainData, validation_data=valData, shuffle=True, epochs=epochs, callbacks=checkpoint)
    return model


def test_predict():
    model = keras.models.load_model('cnn_gost_28x28.h5')
    img = keras.preprocessing.image.load_img('img/2.jpg', target_size=(imageSize, imageSize), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = img_arr/255.
    img_arr = img_arr.reshape((1, imageSize, imageSize, 1))
    result = model.predict_classes(img_arr)

    return result


if __name__ == '__main__':
    model = mnist_cnn_model()
    model = train(model)
    model.save('cnn_gost_28x28.h5')
    # print(test_predict())




