import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import datetime

import os
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
import random
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
    img = cv2.cvtColor(np.asarray(img, dtype='uint8'), cv2.COLOR_RGB2GRAY)
    img = np.expand_dims(img, 2)
    # do crop
    img = img[60:img.shape[0] - 25,:,:]
    # cv2.imwrite("crop.png", img)

    # normalization
    img = (img.astype('float32') / 255.0) - 0.5
    return img


def preprocessing(filepath, center_angle):
    # image preprocessing

    img = cv2.imread(filepath)

    img = cv2.cvtColor(np.asarray(img, dtype='uint8'), cv2.COLOR_BGR2GRAY)  # COLOR_BGR2RGB
    # cv2.imwrite("orig.png", img)
    img = np.expand_dims(img, 2)
    # do crop
    img = img[60:img.shape[0] - 25,:,:]
    # cv2.imwrite("crop.png", cv2.cvtColor(np.asarray(img, dtype='uint8'), cv2.COLOR_BGR2RGB))

    # random flip
    random.seed(0)
    if random.randint(0, 9999) % 2 != 0:
        img = np.fliplr(img)
        center_angle = -center_angle
        # cv2.imwrite("flip.png", cv2.cvtColor(np.asarray(img, dtype='uint8'), cv2.COLOR_BGR2RGB))

    # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # normalization
    img = (img.astype('float32') / 255.0) - 0.5
    return img, center_angle

def generator(_all_xs, _all_ys, batch_size=32):
    num_samples = len(_all_xs)

    all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)
    for offset in range(0, num_samples, batch_size):
        batch_samples = all_xs[offset:offset + batch_size]
        batch_ys = all_ys[offset:offset + batch_size]

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
        yield sklearn.utils.shuffle(X_train, y_train)

# len(train_X) =  19286
# len(test_X) 4822
# conv1.shape =  (?, 71, 316, 24)
# max_pool conv1.shape =  (?, 35, 158, 24)
# conv2.shape =  (?, 31, 154, 36)
# max_pool conv2.shape =  (?, 15, 77, 36)
# conv3.shape =  (?, 11, 73, 48)
# max_pool conv3.shape =  (?, 5, 36, 48)
# conv4.shape =  (?, 3, 34, 64)
# max_pool conv4.shape =  (?, 1, 17, 64)
# fc0.shape =  (?, 1088)
# ____________________________________________________________________________________________________
# best loss = 0.015383704211562871
# Model saved in file: ./session/best_model.ckpt

class bc_network(object):
    def __init__(self):
        # Hyperparameters
        mu = 0
        sigma = 0.1
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, 75, 320, 1], name='input_tensor')
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='labels')
            # SOLUTION: Layer 1: Convolutional. Input = 75x320x1. Output = 71x316x24.
            conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 24), mean=mu, stddev=sigma))
            conv1_b = tf.Variable(tf.zeros(24))
            conv1 = tf.nn.conv2d(self.x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
            print("conv1.shape = ", conv1.get_shape())  # (?, 71, 316, 24)

            # SOLUTION: Activation.
            conv1 = tf.nn.elu(conv1)

            # SOLUTION: Pooling. Input = 71x316x24. Output = 35x158x24.
            self.conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print("max_pool conv1.shape = ", self.conv1.get_shape())  # (?, 35, 158, 24)

            # SOLUTION: Layer 2: Convolutional. Output = 31x154x36.
            conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 36), mean=mu, stddev=sigma))
            conv2_b = tf.Variable(tf.zeros(36))
            conv2 = tf.nn.conv2d(self.conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
            print("conv2.shape = ", conv2.get_shape())  # (?, 31, 154, 36)

            # SOLUTION: Activation.
            conv2 = tf.nn.elu(conv2)

            # SOLUTION: Pooling. Input = 31x154x36. Output = 15x77x36.
            self.conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print("max_pool conv2.shape = ", self.conv2.get_shape())  # (?, 15, 77, 36)

            ####################################################################
            # SOLUTION: Layer 3: Convolutional. Output = 11x73x48.
            conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 36, 48), mean=mu, stddev=sigma))
            conv3_b = tf.Variable(tf.zeros(48))
            conv3 = tf.nn.conv2d(self.conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
            print("conv3.shape = ", conv3.get_shape())  # (?, 11, 73, 48)

            # SOLUTION: Activation.
            conv3 = tf.nn.elu(conv3)

            # SOLUTION: Pooling. Input = 11x73x48. Output = 5x36x48.
            self.conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print("max_pool conv3.shape = ", self.conv3.get_shape())  # (?, 5, 36, 48)

            # SOLUTION: Layer 4: Convolutional. Output = 3x34x64.
            conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 48, 64), mean=mu, stddev=sigma))
            conv4_b = tf.Variable(tf.zeros(64))
            conv4 = tf.nn.conv2d(self.conv3, conv4_W, strides=[1, 1, 1, 1], padding='VALID') + conv4_b
            print("conv4.shape = ", conv4.get_shape())  # (?, 3, 34, 64)

            # SOLUTION: Activation.
            conv4 = tf.nn.elu(conv4)

            # SOLUTION: Pooling. Input = 3x34x64. Output = 1x17x64.
            self.conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            print("max_pool conv4.shape = ", self.conv4.get_shape())  # (?, 1, 17, 64)
            ####################################################################

            # SOLUTION: Flatten. Input = 1x17x64. Output = 1088.
            self.fc0 = flatten(self.conv4)
            print("fc0.shape = ", self.fc0.get_shape())  # (?, 1088)

            # SOLUTION: Layer 5: Fully Connected. Input = 1088. Output = 100.
            fc1_W = tf.Variable(tf.truncated_normal(shape=(1088, 100), mean=mu, stddev=sigma))
            fc1_b = tf.Variable(tf.zeros(100))
            fc1 = tf.matmul(self.fc0, fc1_W) + fc1_b

            # SOLUTION: Activation.
            self.fc1 = tf.nn.elu(fc1)
            self.dropout_layer = tf.nn.dropout(self.fc1, 0.5, name="dropout_layer")

            # SOLUTION: Layer 6: Fully Connected. Input = 100. Output = 50.
            fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 50), mean=mu, stddev=sigma))
            fc2_b = tf.Variable(tf.zeros(50))
            fc2 = tf.matmul(self.dropout_layer, fc2_W) + fc2_b

            # SOLUTION: Activation.
            self.fc2 = tf.nn.elu(fc2)

            # SOLUTION: Layer 7: Fully Connected. Input = 50. Output = 10.
            fc3_W = tf.Variable(tf.truncated_normal(shape=(50, 10), mean=mu, stddev=sigma))
            fc3_b = tf.Variable(tf.zeros(10))
            fc3 = tf.matmul(self.fc2, fc3_W) + fc3_b

            # SOLUTION: Activation.
            self.fc3 = tf.nn.elu(fc3)

            # SOLUTION: Layer 8: Fully Connected. Input = 10. Output = 1.
            fc4_W = tf.Variable(tf.truncated_normal(shape=(10, 1), mean=mu, stddev=sigma))
            fc4_b = tf.Variable(tf.zeros(1))
            self.logits = tf.matmul(self.fc3, fc4_W) + fc4_b

            with tf.name_scope("loss"):
                # MSE损失，将计算值回归到评分
                cost = tf.losses.mean_squared_error(self.y, self.logits)
                self.loss = tf.reduce_mean(cost)
            # 优化损失
            with tf.name_scope('train'):
                self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

            self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

            self.save_dir = './session'
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)
            self.ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if self.ckpt and self.ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
                print("Successfully loaded:", tf.train.latest_checkpoint(self.save_dir))
            else:
                print("Could not find old network weights")


    def training(self, xs, labels, ii, epoch_i, batch_i, batch_num):
        feed_dict = {
            self.x: xs,
            self.y: labels
        }

        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        if (ii % 20 == 0):
            time_str = datetime.datetime.now().isoformat()
            print('Training {}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.5f}'.format(
                time_str,
                epoch_i,
                batch_i,
                batch_num,
                loss))

        return loss

    def testing(self, xs, labels, ii, epoch_i, batch_i, batch_num):
        feed_dict = {
            self.x: xs,
            self.y: labels
        }

        loss = self.sess.run([self.loss], feed_dict=feed_dict)
        loss = loss[0]
        if (ii % 20 == 0):
            time_str = datetime.datetime.now().isoformat()
            print('#Testing# {}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.5f}'.format(
                time_str,
                epoch_i,
                batch_i,
                batch_num,
                loss))

        return loss

    def save(self):
        save_path = self.saver.save(self.sess, os.path.join(self.save_dir, 'best_model.ckpt'))
        print("Model saved in file: {}".format(save_path))

    def forward(self, xs):
        feed_dict = {
            self.x: xs
        }
        logits = self.sess.run([self.logits], feed_dict=feed_dict)

        return logits[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    args = parser.parse_args()

    # read csv file dump to local file
    prepare_data()
    train_X, test_X, train_y, test_y = get_dataset()
    print("len(train_X) = ", len(train_X))
    print("len(test_X)", len(test_X))

    # compile and train the model using the generator function
    net = bc_network()

    epochs = 3

    best_loss = 9999
    for ii in range(epochs):
        train_batches = generator(train_X, train_y, 32)
        test_batches = generator(test_X, test_y, 32)
        batch_num = (len(train_X) // 32)
        for batch_i in range(batch_num):
            x, y = next(train_batches)
            net.training(x, y, ii * (batch_num) + batch_i, ii, batch_i, batch_num)

        batch_num = (len(test_X) // 32)
        test_loss = 0.0

        for batch_i in range(batch_num):
            x, y = next(test_batches)
            loss = net.testing(x, y, ii * (batch_num) + batch_i, ii, batch_i, batch_num)
            test_loss = test_loss + loss

        test_loss = test_loss / batch_num
        if test_loss < best_loss:
            best_loss = test_loss
            print("best loss = {}".format(best_loss))
            net.save()
        else:
            print("test loss = {}".format(test_loss))


