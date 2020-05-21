##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Haihao Zhu
## ShanghaiTech University
## zhuhh2@shanghaitech.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import tensorflow as tf
import os
from tensorflow import keras
from utils.config import opt

from models.TSCNet import TSCNet
from models.ResNet import *
from models.FCNLSTM import build_fcnlstm
from models.FCNGRU import build_fcngru
from models.FCN import build_fcn
from models.FCNALSTM import build_fcnalstm
from models.FCNABLSTM import build_fcnablstm
from models.FCNABLSTM2 import build_fcnablstm2
from utils.data_loader import load_data
from utils.model_solver import Solver
import argparse
import logging
from pprint import pprint

from sklearn.preprocessing import LabelBinarizer

# logger config
logger = logging.getLogger('TSC')
logger.setLevel(logging.INFO)
#fh = logging.FileHandler('./results/log.txt', mode='w')
sh = logging.StreamHandler()
#fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)
fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
formatter = logging.Formatter(fmt)
#fh.setFormatter(formatter)
sh.setFormatter(formatter)
#logger.addHandler(fh)
logger.addHandler(sh)

# train model from scratch
def train_scratch(model_name):
    result_scratch_file = './results/result_scratch.csv'
    result_finetune_file = './results/result_finetune.csv'

    with open(result_scratch_file, 'a') as f:
        f.seek(0)
        f.truncate()
        f.write('dataset_name,{0}\n'.format(model_name))

    with open(result_finetune_file, 'a') as f:
        f.seek(0)
        f.truncate()
        f.write('dataset_name,{0}\n'.format(model_name))

    for dataset_name in opt.dataset.test_dataset_names_44:
        optimizer = getattr(keras.optimizers, opt.train.optimizer)()
        criterion = keras.losses.CategoricalCrossentropy()
        metric = keras.metrics.CategoricalAccuracy()
        # val_metric = keras.metrics.CategoricalAccuracy()

        lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor=opt.train.monitor, factor=opt.train.lr_factor,
                                                         patience=opt.train.lr_patience, min_lr=opt.train.lr_min_lr,
                                                         verbose=1)
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=opt.train.checkpoint_path + '_' + dataset_name,
                                                     save_best_only=True,
                                                     mode='max',
                                                     save_weights_only=True, monitor='val_categorical_accuracy',
                                                     verbose=1)
        tensorboard = keras.callbacks.TensorBoard(log_dir=opt.train.log_dir, update_freq='epoch')
        csv_logger = keras.callbacks.CSVLogger('results/logs/training.log')
        early_stop = keras.callbacks.EarlyStopping(monitor=opt.train.monitor, min_delta=opt.train.stop_min_delta,
                                                   patience=opt.train.stop_patience, verbose=1,
                                                   restore_best_weights=True)
        # callbacks = [lr_scheduler, tensorboard, csv_logger, early_stop]
        callbacks = [lr_scheduler, early_stop, checkpoint]


        # get data
        logger.info('============loading data {0}============'.format(dataset_name))
        train_data, test_data, input_shape, num_classes, batch_size = load_data(opt, dataset_name)
        #lb = LabelBinarizer()
        #y_train_onehot = lb.fit_transform(y_train)
        logger.info(('===========Done==============='))
        if model_name != opt.model.name:
            opt.model.name = model_name
        # get model
        model = None
        if opt.model.name == 'TSCNet':
            model = TSCNet(input_shape, num_classes, opt.model.num_layers)
        elif opt.model.name == 'ResNet':
            x, y = build_resnet(input_shape, 64, num_classes)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == 'FCN':
            x, y = build_fcn(input_shape, num_classes)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == 'ResNet10':
            x, y = build_resnet10(input_shape, num_classes)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == 'ResNet18':
            x, y = build_resnet18(input_shape, num_classes)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == 'ResNet34':
            x, y = build_resnet34(input_shape, num_classes)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == 'ResNet50':
            x, y = build_resnet50(input_shape, num_classes)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == "FCNLSTM":
            x, y = build_fcnlstm(input_shape, num_classes, num_cells=8)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == "FCNGRU":
            x, y = build_fcngru(input_shape, num_classes, num_cells=8)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == "FCNALSTM":
            x, y = build_fcnalstm(input_shape, num_classes, num_cells=8)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == "FCNABLSTM":
            x, y = build_fcnablstm(input_shape, num_classes, num_cells=64, dropout_rate=0.3, embedding_size=64)
            model = keras.models.Model(inputs=x, outputs=y)
        elif opt.model.name == "FCNABLSTM2":
            x, y = build_fcnablstm2(input_shape, num_classes, num_cells=64, dropout_rate=0.3, embedding_size=64)
            model = keras.models.Model(inputs=x, outputs=y)
        # summary model
        print('model builed!')
        if model:
            model.summary()
        solver = Solver(opt, model, dataset_name, num_classes)
        #batch_size = int(min(train_data.shape[0] / 10, opt.train.batch_size))
        trd, ted = train_data.batch(batch_size), test_data.batch(batch_size)
        solver.fit(train_data=trd, test_data=ted, optimizer=optimizer, criterion=criterion,
                   callbacks=callbacks, metric=metric)
        solver.load(opt.train.checkpoint_path+'_'+dataset_name)
        _, acc = solver.evaluate(ted)

        with open(result_scratch_file, 'a') as f:
            f.write('{0},{1}\n'.format(dataset_name, acc))
        """
        # fine-tune model here
        print("===============fine-tune start===============")
        initial_lr = 0.001
        initial_bs = 128
        for i in range(5):
            if initial_lr < 0.0001 or initial_bs < 32:
                break
            if i != 0:
                model.load_weights(opt.ft.modelweights_path+'_'+dataset_name)
            else:
                model.load_weights(opt.train.checkpoint_path+'_'+dataset_name)
            optimizer = keras.optimizers.Adam(lr=initial_lr)
            solver = Solver(opt, model, dataset_name, num_classes)
            trd, ted = train_data.batch(initial_bs), test_data.batch(initial_bs)
            checkpoint = keras.callbacks.ModelCheckpoint(filepath=opt.ft.modelweights_path + '_' + dataset_name, save_best_only=True,
                                                         mode='max',
                                                         save_weights_only=True, monitor='val_categorical_accuracy', verbose=1)
            callbacks = [lr_scheduler, early_stop, checkpoint]
            solver.fit(train_data=trd, test_data=ted,
                       optimizer=optimizer, criterion=criterion,
                       callbacks=callbacks, metric=metric)

            initial_lr /= 2
            initial_bs //= 2

        print("=========after fine tune:==========")
        model.load_weights(opt.ft.modelweights_path + '_' + dataset_name)
        solver = Solver(opt, model, dataset_name, num_classes)
        _, acc = solver.evaluate(ted)
        with open(result_finetune_file, 'a') as f:
            f.write('{0},{1}\n'.format(dataset_name, acc))

        """


# pre-train model for transfer learning
def train_pre():
    pass


# fine-tune model
def train_transfer():
    pass


def main():
    logging.info('=======config info======')
    logging.info(pprint(opt))
    logging.info('======= end ======')

    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--config_file', type=str, default=None, help='optional config file for training')
    parser.add_argument('--exp_type', type=str, default='scratch', help='experiment type')
    parser.add_argument('--model', type=str, default='TSCNet', help='model name')

    args = parser.parse_args()

    if args.exp_type == 'scratch':
        train_scratch(model_name=args.model)
    elif args.exp_type == 'pre':
        train_pre()
    elif args.exp_type == 'transfer':
        train_transfer()
    else:
        logger.info('valid exp_type: scratch/pre/transfer')



if __name__ == '__main__':
    main()