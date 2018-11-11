import cv2
import numpy as np
# import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
import random
import time
import argparse


def prepare_data():
    filepaths = []
    steerings = []
    samples = []
    path = "./data"
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[3] == "steering":
                continue
            # samples.append(line)
            steering_center = float(line[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2  # this is a parameter to tune  0.1
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            filepaths.append(os.path.join(path, line[0]))
            filepaths.append(os.path.join(path, line[1]))
            filepaths.append(os.path.join(path, line[2]))

            steerings.append(steering_center)
            steerings.append(steering_left)
            steerings.append(steering_right)

    train_X, test_X, train_y, test_y = train_test_split(filepaths,
                                                        steerings,
                                                        test_size=0.2,
                                                        random_state=0)

    pickle.dump(train_X, open('train_X.p', 'wb'))
    pickle.dump(train_y, open('train_y.p', 'wb'))
    pickle.dump(test_X, open('test_X.p', 'wb'))
    pickle.dump(test_y, open('test_y.p', 'wb'))


def get_dataset():
    train_X = pickle.load(open('train_X.p', mode='rb'))
    train_y = pickle.load(open('train_y.p', mode='rb'))
    test_X = pickle.load(open('test_X.p', mode='rb'))
    test_y = pickle.load(open('test_y.p', mode='rb'))
    return train_X, test_X, train_y, test_y


def preprocessing_drive(image_array):
    # print(filepath)

    # img = cv2.imread(filepath)
    # img = cv2.cvtColor(np.asarray(img, dtype='uint8'), cv2.COLOR_BGR2RGB)
    # cv2.imwrite("orig.png", img)
    img = image_array
    img = img[60:img.shape[0] - 25,:,:]
    # cv2.imwrite("crop.png", img)

    # random.seed(0)
    # if random.randint(0, 9999) % 2 != 0:
    #     img = np.fliplr(img)
        # cv2.imwrite("flip.png", img)
    # else:
    #     pass
    # img = cv2.resize(img, (64, 64))
    img = (img.astype('float32') / 255.0) - 0.5
    return img


def preprocessing(filepath, center_angle):
    # print(filepath)

    img = cv2.imread(filepath)
    img = cv2.cvtColor(np.asarray(img, dtype='uint8'), cv2.COLOR_BGR2RGB)
    # cv2.imwrite("orig.png", img)
    img = img[60:img.shape[0] - 25,:,:]
    cv2.imwrite("crop.png", img)

    random.seed(0)
    if random.randint(0, 9999) % 2 != 0:
        img = np.fliplr(img)
        center_angle = -center_angle
        # cv2.imwrite("flip.png", img)
    # else:
    #     pass
    # img = cv2.resize(img, (64, 64))
    img = (img.astype('float32') / 255.0) - 0.5
    return img, center_angle



def generator(_all_xs, _all_ys, batch_size=32):
    num_samples = len(_all_xs)
    while True: # Loop forever so the generator never terminates
        # shuffle(samples)
        all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)
        for offset in range(0, num_samples, batch_size):
            batch_samples = all_xs[offset:offset+batch_size]
            batch_ys = all_ys[offset:offset+batch_size]

            images = []
            angles = []
            for i, filepath in enumerate(batch_samples):
                # name = batch_sample  # './IMG/'+batch_sample[0].split('/')[-1]
                center_image, center_angle = preprocessing(filepath, batch_ys[i])  # cv2.imread(name)
                # center_angle = batch_ys[i]  # float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)  #

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# len(train_X) =  19286
# len(test_X) 4822
# ____________________________________________________________________________________________________
# Layer (type)                     Output Shape          Param #     Connected to
# ====================================================================================================
# convolution2d_1 (Convolution2D)  (None, 75, 320, 1)    4           convolution2d_input_1[0][0]
# ____________________________________________________________________________________________________
# convolution2d_2 (Convolution2D)  (None, 71, 316, 24)   624         convolution2d_1[0][0]
# ____________________________________________________________________________________________________
# maxpooling2d_1 (MaxPooling2D)    (None, 35, 158, 24)   0           convolution2d_2[0][0]
# ____________________________________________________________________________________________________
# convolution2d_3 (Convolution2D)  (None, 31, 154, 36)   21636       maxpooling2d_1[0][0]
# ____________________________________________________________________________________________________
# maxpooling2d_2 (MaxPooling2D)    (None, 15, 77, 36)    0           convolution2d_3[0][0]
# ____________________________________________________________________________________________________
# convolution2d_4 (Convolution2D)  (None, 11, 73, 48)    43248       maxpooling2d_2[0][0]
# ____________________________________________________________________________________________________
# maxpooling2d_3 (MaxPooling2D)    (None, 5, 36, 48)     0           convolution2d_4[0][0]
# ____________________________________________________________________________________________________
# convolution2d_5 (Convolution2D)  (None, 3, 34, 64)     27712       maxpooling2d_3[0][0]
# ____________________________________________________________________________________________________
# maxpooling2d_4 (MaxPooling2D)    (None, 1, 17, 64)     0           convolution2d_5[0][0]
# ____________________________________________________________________________________________________
# flatten_1 (Flatten)              (None, 1088)          0           maxpooling2d_4[0][0]
# ____________________________________________________________________________________________________
# dense_1 (Dense)                  (None, 100)           108900      flatten_1[0][0]
# ____________________________________________________________________________________________________
# dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]
# ____________________________________________________________________________________________________
# dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]
# ____________________________________________________________________________________________________
# dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
# ____________________________________________________________________________________________________
# dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
# ====================================================================================================
# Total params: 207,695
# Trainable params: 207,695
# Non-trainable params: 0
# ____________________________________________________________________________________________________
# WARNING:tensorflow:From /Applications/anaconda/envs/carnd-term1/lib/python3.5/site-packages/keras/callbacks.py:618 in set_model.: merge_all_summaries (from tensorflow.python.ops.logging_ops) is deprecated and will be removed after 2016-11-30.
# Instructions for updating:
# Please switch to tf.summary.merge_all.
# Epoch 1/10
# 19264/19286 [============================>.] - ETA: 1s - loss: 0.0205Epoch 00000: val_loss improved from inf to 0.01735, saving model to model.h5
# 19286/19286 [==============================] - 1247s - loss: 0.0205 - val_loss: 0.0174
# Epoch 2/10
# 19264/19286 [============================>.] - ETA: 1s - loss: 0.0165Epoch 00001: val_loss improved from 0.01735 to 0.01486, saving model to model.h5
# 19286/19286 [==============================] - 1197s - loss: 0.0165 - val_loss: 0.0149
# Epoch 3/10
# 19264/19286 [============================>.] - ETA: 1s - loss: 0.0145Epoch 00002: val_loss improved from 0.01486 to 0.01441, saving model to model.h5
# 19286/19286 [==============================] - 1193s - loss: 0.0145 - val_loss: 0.0144
# Epoch 4/10
# 19264/19286 [============================>.] - ETA: 1s - loss: 0.0127Epoch 00003: val_loss did not improve
# 19286/19286 [==============================] - 1177s - loss: 0.0127 - val_loss: 0.0146
# Epoch 5/10
# 19264/19286 [============================>.] - ETA: 1s - loss: 0.0115Epoch 00004: val_loss improved from 0.01441 to 0.01348, saving model to model.h5


class bc_network(object):
    def __init__(self):

        #
        # ch, row, col = 3, 64, 64  # Trimmed image format
        #
        # if K.image_data_format() == 'channels_first':
        #     input_shape = (3, 64, 64)
        # else:
        input_shape = (75, 320, 3)  # (64, 64, 3)

        self.model = Sequential()

        self.model.add(Conv2D(1, 1, 1, border_mode='same', input_shape=input_shape))
        self.model.add(Conv2D(24, 5, 5, activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(36, 5, 5, activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(48, 5, 5, activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, 3, 3, activation='elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # self.model.add(Conv2D(64, 3, 3, activation='elu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='elu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(50, activation='elu'))
        self.model.add(Dense(10, activation='elu'))

        self.model.add(Dense(1))

        self.model.summary()

        self.model.compile(loss='mse', optimizer='adam')
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
        #                       metrics=['accuracy'])
        # model.compile(loss=keras.losses.categorical_crossentropy,
        #               optimizer=keras.optimizers.Adadelta(),
        #               metrics=['accuracy'])

        # train the model
        self.checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1,
                                       save_best_only=True)
        # model.save('model.h5')
        self.weights = 'model.h5'
        if (os.path.exists(self.weights)):
            self.model.load_weights('model.h5')
            print("model loaded!")

        # datagen_train.fit(train_xs)


    def train(self, train_generator, validation_generator, samples_per_epoch, nb_val_samples, epochs=1):

        # timestamp = str(int(time.time()))
        out_dir="./run"
        if (os.path.exists(out_dir)):
            # shutil.rmtree("/Users/bitware/Desktop/test/subimg")
            pass
        else:
            os.mkdir(out_dir)
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        # print("Writing to {}\n".format(out_dir))
        #
        # model.fit_generator(datagen_train.flow(train_xs, train_ys, batch_size=batch_size),
        #                     steps_per_epoch=train_xs.shape[0] // batch_size,
        #                     epochs=epochs, verbose=2, callbacks=[checkpointer, TensorBoard(log_dir=out_dir)],
        #                     validation_data=(test_xs, test_ys),
        #                     validation_steps=test_xs.shape[0] // batch_size)

        # model.compile(loss='mse', optimizer='adam')
        history_object = self.model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                                 validation_data=validation_generator, nb_val_samples=nb_val_samples,
                                 nb_epoch=epochs, callbacks=[self.checkpointer, TensorBoard(log_dir=out_dir)])
        #
        # """
        # If the above code throw exceptions, try
        # model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
        # validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
        # """
        #
        #
        return history_object


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, help='epochs')
    # parser.add_argument('--file', default="sample1_300.png", type=str, help='filename')
    # correction("/Users/bitware/Desktop/test", "sample1_300.png")
    args = parser.parse_args()
    # prepare_data()
    train_X, test_X, train_y, test_y = get_dataset()
    print("len(train_X) = ", len(train_X))
    print("len(test_X)", len(test_X))
    # compile and train the model using the generator function
    train_generator = generator(train_X, train_y, batch_size=32)
    validation_generator = generator(test_X, test_y, batch_size=32)

    n = random.randint(0, len(train_X) - 1)
    preprocessing(train_X[n], train_y[n])

    net = bc_network()
    history_object = net.train(train_generator, validation_generator, len(train_X), len(test_X), args.epochs)




    #
    # history_object = model.fit_generator(train_generator, samples_per_epoch =
    #     len(train_samples), validation_data =
    #     validation_generator,
    #     nb_val_samples = len(validation_samples),
    #     nb_epoch=5, verbose=1)
    #
    ### print the keys contained in the history object
    # print(history_object.history.keys())
    #
    # ### plot the training and validation loss for each epoch
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()
    #



