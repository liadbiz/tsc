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
import tensorflow.keras as keras
from utils.config import opt

import models
from utils.data_loader import load_data
from utils.model_solver import Solver
import argparse
import logging
from pprint import pprint


logger = logging.getLogger('TSC')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('./logs/tsc.log', mode='w')
sh = logging.StreamHandler()
fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)
fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
formatter = logging.Formatter(fmt)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# train model from scratch
def train_scratch():
    optimizer = keras.optimizers.Adam(learning_rate=opt.train.lr)
    criterion = keras.losses.CategoricalCrossentropy(from_logits=True)
    train_metric = keras.metrics.CategoricalAccuracy()
    # val_metric = keras.metrics.CategoricalAccuracy()

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor=opt.train.monitor, factor=opt.train.factor,
                                                     patience=opt.train.patience, min_lr=opt.train.min_lr)

    for dataset_name in opt.dataset.test_dataset_names:
        # get data
        logger.info('============loading data============')
        train_data, test_data, input_shape, num_classes = load_data(opt, dataset_name)
        logger.info(('===========Done==============='))

        # get model
        model = getattr(models, opt.model.name)(input_shape, num_classes, opt.model.num_layers)
        # summary model
        model.build(input_shape=keras.Input(shape=input_shape).shape)
        model.summary()
        solver = Solver(opt, model, dataset_name, num_classes)
        solver.fit(train_data=train_data, test_data=test_data, optimizer=optimizer, criterion=criterion,
                   lr_scheduler=lr_scheduler, train_metric=train_metric)

# pre-train model for transfer learning
def train_pre():
    pass

# fine-tune model
def train_transfer():
    pass

def main():
    # logger config
    logger = logging.getLogger('TSC')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('./logs/tsc.log', mode='w')
    sh = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logging.info('=======config info======')
    logging.info(pprint(opt))
    logging.info('======= end ======')

    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--config_file', type=str, default=None, help='optional config file for training')
    parser.add_argument('--exp_type', type=str, default='scratch', help='experiment type')

    args = parser.parse_args()

    if args.exp_type == 'scratch':
        train_scratch()
    elif args.exp_type == 'pre':
        train_pre()
    elif args.exp_type == 'transfer':
        train_transfer()
    else:
        logger.info('valid exp_type: scratch/pre/transfer')



if __name__ == '__main__':
    main()