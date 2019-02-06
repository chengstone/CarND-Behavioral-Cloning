import cv2
import numpy as np
# import tensorflow as tf
# from keras.models import Sequential, Model
# from keras.layers import Lambda
# from keras.layers import Cropping2D
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D, Convolution2D
# from keras import backend as K
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import TensorBoard
# from keras.optimizers import Adam
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
    # read csv file dump to local file
    filepaths = []
    steerings = []
    path = "./data"
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[3] == "steering":
                continue
            steering_center = float(line[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2  # this is a parameter to tune
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
    # load dataset from local file

    train_X = pickle.load(open('train_X.p', mode='rb'))
    train_y = pickle.load(open('train_y.p', mode='rb'))
    test_X = pickle.load(open('test_X.p', mode='rb'))
    test_y = pickle.load(open('test_y.p', mode='rb'))
    return train_X, test_X, train_y, test_y


def preprocessing_drive(image_array):
    # provide for drive.py

    img = image_array

    # do crop
    img = img[60:img.shape[0] - 25,:,:]
    # cv2.imwrite("crop.png", img)

    # normalization
    img = (img.astype('float32') / 255.0) - 0.5
    return img


def preprocessing(filepath, center_angle):
    # image preprocessing

    img = cv2.imread(filepath)
    img = cv2.cvtColor(np.asarray(img, dtype='uint8'), cv2.COLOR_BGR2RGB)
    # cv2.imwrite("orig.png", img)

    # do crop
    img = img[60:img.shape[0] - 25,:,:]
    # cv2.imwrite("crop.png", img)

    # random flip
    random.seed(0)
    if random.randint(0, 9999) % 2 != 0:
        img = np.fliplr(img)
        center_angle = -center_angle
        # cv2.imwrite("flip.png", img)

    # normalization
    img = (img.astype('float32') / 255.0) - 0.5
    return img, center_angle


def generator(_all_xs, _all_ys, batch_size=32):
    num_samples = len(_all_xs)
    if True: # Loop forever so the generator never terminates
        all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)
        for offset in range(0, num_samples, batch_size):
            batch_samples = all_xs[offset:offset+batch_size]
            batch_ys = all_ys[offset:offset+batch_size]

            images = []
            angles = []
            for i, filepath in enumerate(batch_samples):
                # do preprocessing for every image and return the angle also
                center_image, center_angle = preprocessing(filepath, batch_ys[i])

                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            y_train = np.expand_dims(y_train, 1)
            yield sklearn.utils.shuffle(X_train.astype(np.float32), y_train.astype(np.float32))

# len(train_X) =  19286
# len(test_X) 4822
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 75, 320, 1)        4
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 71, 316, 24)       624
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 35, 158, 24)       0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 31, 154, 36)       21636
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 15, 77, 36)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 11, 73, 48)        43248
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 5, 36, 48)         0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 3, 34, 64)         27712
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 1, 17, 64)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 1088)              0
# _________________________________________________________________
# dense (Dense)                (None, 100)               108900
# _________________________________________________________________
# dropout (Dropout)            (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 50)                5050
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                510
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 207,695
# Trainable params: 207,695
# Non-trainable params: 0
# _________________________________________________________________
#
# Model test set loss: 0.0174 mae: 0.1006
# best loss = 0.017396364361047745
# Model test set loss: 0.0148 mae: 0.0911
# best loss = 0.014845996163785458
# Model test set loss: 0.0147 mae: 0.0911
# best loss = 0.014654010534286499
# Model test set loss: 0.014475 mae: 0.089655
# best loss = 0.014475451782345772
# Model test set loss: 0.014029 mae: 0.087233
# best loss = 0.014029275625944138


import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
import time

MODEL_DIR = "./models"


class bc_network(object):
    def __init__(self):
        self.batch_size = 32
        self.best_loss = 9999
        input_shape = (75, 320, 3)
        # Hyperparameters
        self.model = keras.Sequential([
            keras.layers.Conv2D(1, 1, 1, padding='same', input_shape=input_shape),
            keras.layers.Conv2D(24, 5, 1, activation='elu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(36, 5, 1, activation='elu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(48, 5, 1, activation='elu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, 3, 1, activation='elu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='elu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(50, activation='elu'),
            keras.layers.Dense(10, activation='elu'),
            keras.layers.Dense(1)
        ])
        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam()
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

        if tf.io.gfile.exists(MODEL_DIR):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
        test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')

        self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
        self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # @tf.function
    def compute_loss(self, labels, logits):
        return tf.reduce_mean(tf.keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        return tf.keras.metrics.mae(labels, logits)  #

    @tf.function
    def train_step(self, images, labels):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss = self.ComputeLoss(labels, logits)
            # loss = self.compute_loss(labels, logits)
            self.ComputeMetrics(labels, logits)
            # metrics = self.compute_metrics(labels, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics, logits

    def training(self, train_dataset, test_dataset, epochs=10, log_freq=50):

        train_X, train_y = train_dataset

        for i in range(epochs):
            train_batches = generator(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // 32)

            train_start = time.time()
            with self.train_summary_writer.as_default():
                start = time.time()
                # Metrics are stateful. They accumulate values and return a cumulative
                # result when you call .result(). Clear accumulated values with .reset_states()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

                # Datasets can be iterated over like any other Python iterable.
                for batch_i in range(batch_num):
                    images, labels = next(train_batches)
                # for images, labels in train_dataset:
                    loss, metrics, logits = self.train_step(images, labels)
                    avg_loss(loss)
                    avg_mae(metrics)

                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        summary_ops_v2.scalar('loss', avg_loss.result(), step=self.optimizer.iterations)
                        summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=self.optimizer.iterations)
                        # summary_ops_v2.scalar('mae', avg_mae.result(), step=self.optimizer.iterations)

                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(), loss, (self.ComputeMetrics.result()), rate))
                        # print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                        #     self.optimizer.iterations.numpy(), loss, (avg_mae.result()), rate))
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        # avg_mae.reset_states()
                        start = time.time()

            train_end = time.time()
            print('\nTrain time for epoch #{} ({} total steps): {}'.format(i + 1, self.optimizer.iterations.numpy(),
                                                                           train_end - train_start))
            with self.test_summary_writer.as_default():
                self.testing(test_dataset, self.optimizer.iterations)
            # self.checkpoint.save(self.checkpoint_prefix)
        self.export_path = os.path.join(MODEL_DIR, 'export')
        tf.saved_model.save(self.model, self.export_path)

    def testing(self, test_dataset, step_num):
        test_X, test_y = test_dataset
        test_batches = generator(test_X, test_y, self.batch_size)

        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

        batch_num = (len(test_X) // 32)
        for batch_i in range(batch_num):
            images, labels = next(test_batches)
        # for (images, labels) in test_dataset:
            logits = self.model(images, training=False)
            avg_loss(self.ComputeLoss(labels, logits))
            self.ComputeMetrics(labels, logits)
            # avg_loss(self.compute_loss(labels, logits))
            # avg_mae(self.compute_metrics(labels, logits))

        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), self.ComputeMetrics.result()))
        # print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), avg_mae.result()))
        summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
        summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=step_num)
        # summary_ops_v2.scalar('mae', avg_mae.result(), step=step_num)

        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            print("best loss = {}".format(self.best_loss))
            self.checkpoint.save(self.checkpoint_prefix)

    # def evaluating(self, test_dataset):
    #     #         restored_model = tf.saved_model.restore(self.export_path)
    #     #         y_predict = restored_model(x_test)
    #     avg_accuracy = tf.keras.metrics.Mean('accuracy', dtype=tf.float32)
    #
    #     for (images, labels) in test_dataset:
    #         logits = self.model(images, training=False)
    #         avg_accuracy(self.compute_accuracy(logits, labels))
    #
    #     print('Model accuracy: {:0.2f}%'.format(avg_accuracy.result() * 100))

    def forward(self, xs):
        predictions = self.model(xs)
        # logits = tf.nn.softmax(predictions)

        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=3, type=int, help='epochs')
    args = parser.parse_args()

    # read csv file dump to local file
    # prepare_data()
    train_X, test_X, train_y, test_y = get_dataset()
    print("len(train_X) = ", len(train_X))
    print("len(test_X)", len(test_X))

    # train_generator = generator(train_X, train_y, batch_size=32)
    # validation_generator = generator(test_X, test_y, batch_size=32)

    # test code
    # n = random.randint(0, len(train_X) - 1)
    # preprocessing(train_X[n], train_y[n])

    # compile and train the model using the generator function
    net = bc_network()
    net.training((train_X, train_y), (test_X, test_y), args.epochs)




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



