##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Haihao Zhu
## ShanghaiTech University
## zhuhh2@shanghaitech.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import yaml
from easydict import EasyDict as edict

__C = edict()
opt = __C
__C.seed = 0

__C.dataset = edict()
__C.dataset.archive_name = 'TSC'
# change here to your path where you place UCR datasets. The path should contain a directory named 'TSC' which
# contain 88 dataset's subdirectory.
__C.dataset.dataset_path = '../dataset/'
# not all 88 dataset will be used, classification accuracy will only be evaluated on
# dataset below because other algorithms compared did not give all results on all 88 datasets.
# test_dataset_names: evaluate
# train_dataset_names: pre-train
# valid_dataset_names: validate
__C.dataset.test_dataset_names = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
                            'ChlorineConcentration', 'CinCECGTorso', 'Coffee',
                            'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW',
                            'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour',
                            'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines',
                            'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
                            'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages',
                            'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW',
                            'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil',
                            'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
                            'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1',
                            'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols',
                            'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
                            'TwoPatterns', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
                            'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']
__C.dataset.test_dataset_names = ['TwoPatterns']
__C.model = edict()
__C.model.name = "TSCNet"
__C.model.num_layers = 5

__C.train = edict()
__C.train.optimizer = 'SGD'
__C.train.lr = 0.01
__C.train.lr_factor = 0.2
__C.train.lr_patience = 50
__C.train.lr_min_lr = 0.0001
__C.train.lr_min_delta = 0.001
__C.train.monitor = 'val_categorical_accuracy'
__C.train.stop_patience = 200
__C.train.stop_min_delta = 0.0001
__C.train.num_epochs = 2000
__C.train.batch_size = 128
__C.train.gpus = '0'