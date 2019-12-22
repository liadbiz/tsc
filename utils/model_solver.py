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
import time
class Solver(object):
    def __init__(self, opt, model, dataset_name, num_classes):
        self.opt = opt
        self.model = model
        self.dataset_name = dataset_name
        self.num_classes = num_classes

    def fit(self, train_data, test_data, optimizer, criterion, lr_scheduler, train_metric):
        # train
        start_time = time.time()
        for ep in range(self.opt.train.num_epochs):
            for step, (x, y) in enumerate(train_data):
                #print(y.shape)
                with tf.GradientTape() as tape:
                    logits = self.model(x)
                    loss = criterion(tf.one_hot(y, depth=self.num_classes), logits)
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.train_variables))
                train_metric.update_state(y, logits)
                if ep % 100 == 0:
                    print('loss of epoch', ep, 'is ', loss)
                    print('acc of epoch ', ep, 'is ', train_metric.result().numpy())

            # val
            # TODO
        end_time = time.time()
        duration = end_time - start_time
        print('duration of training dataset {0} is {1}'.format(self.dataset_name, duration))


    def save(self):
        pass

    def predict(self):
        pass

    def test(self):
        pass