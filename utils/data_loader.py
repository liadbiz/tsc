##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Haihao Zhu
## ShanghaiTech University
## zhuhh2@shanghaitech.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from builtins import print
import numpy as np
import pandas as pd
import matplotlib
import random
from utils.config import opt
import tensorflow as tf
from tensorflow import keras
matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import os
import operator
import utils



def load_data(opt, dataset_name):
    # get train/test data
    data_path = opt.dataset.dataset_path + opt.dataset.archive_name + '/' + dataset_name + '/'
    x_train, y_train = readucr(data_path + dataset_name + '_TRAIN.txt')
    x_test, y_test = readucr(data_path + dataset_name + '_TEST.txt')

    assert len(np.unique(y_test)) == len(np.unique(y_train))
    num_classes = len(np.unique(y_test))

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (num_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (num_classes - 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train, x_test = np.expand_dims(x_train, axis=2), np.expand_dims(x_test, axis=2)
    input_shape = x_train.shape[1:]
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    batch_size = int(min(x_train.shape[0] / 10, opt.train.batch_size))
    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1000)
    data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size=1000)

    return data_train, data_test, input_shape, num_classes, batch_size
    #return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def readucr(filename):
    data = np.loadtxt(filename)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def readsits(filename):
    data = np.loadtxt(filename, delimiter=delimiter)
    Y = data[:, -1]
    X = data[:, :-1]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def read_datasets(root_dir, archive_name, dataset_names):

    datasets_dict = {}
    for dataset_name  in dataset_names:
        file_name = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = readucr(file_name + '_TRAIN.txt')
        x_test, y_test = readucr(file_name + '_TEST.txt')
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    return datasets_dict


def read_all_datasets(root_dir, archive_name):
    datasets_dict = {}

    dataset_names_to_sort = []

    if archive_name == 'TSC':
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN.txt')
            x_test, y_test = readucr(file_name + '_TEST.txt')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    elif archive_name == 'InlineSkateXPs':

        for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
            root_dir_dataset = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'SITS':
        return read_sits_xps(root_dir)
    else:
        print('error in archive name')
        exit()

    return datasets_dict


def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_train_test)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test)
    # resplit the train and test
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]
    return new_y_train, new_y_test


def generate_results_csv(output_file_name, root_dir, clfs):
    res = pd.DataFrame(data=np.zeros((0, 8), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name', 'iteration',
                                'precision', 'accuracy', 'recall', 'duration'])
    for archive_name in ARCHIVE_NAMES:
        datasets_dict = read_all_datasets(root_dir, archive_name)
        for classifier_name in clfs:
            durr = 0.0

            curr_archive_name = archive_name
            for dataset_name in datasets_dict.keys():
                output_dir = root_dir + '/results/' + classifier_name + '/' \
                             + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                print(output_dir)
                if not os.path.exists(output_dir):
                    continue
                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['archive_name'] = archive_name
                df_metrics['dataset_name'] = dataset_name
                df_metrics['iteration'] = 0
                res = pd.concat((res, df_metrics), axis=0, sort=False)
                durr += df_metrics['duration'][0]

    res.to_csv(root_dir + output_file_name, index=False)

    res = res.loc[res['classifier_name'].isin(clfs)]

    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration,
              lr=True, plot_test_acc=True):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    if plot_test_acc:
        df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    if plot_test_acc:
        df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    if plot_test_acc:
        # plot losses
        plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def generate_array_of_colors(n):
    # https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    alpha = 1.0
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r / 255, g / 255, b / 255, alpha))
    return ret


def read_sits_xps(root_dir):
    datasets_dict = {}
    path_to_data = root_dir + 'archives/SITS/resampled-SITS/'
    path_to_test = root_dir + 'archives/SITS/' + 'SatelliteFull_TEST_1000.csv'

    x_test, y_test = readsits(path_to_test)

    for subdir, dirs, files in os.walk(path_to_data):
        for file_name in files:
            arr = file_name.split('.')
            dataset_name = arr[0]
            file_type = arr[1]
            if file_type == 'csv':
                x_train, y_train = readsits(subdir + '/' + file_name)

                datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                               y_test.copy())

    return datasets_dict


def resample_dataset(x, rate):
    new_x = np.zeros(shape=(x.shape[0], rate))
    from scipy import signal
    for i in range(x.shape[0]):
        f = signal.resample(x[0], rate)
        new_x[i] = f
    return new_x


