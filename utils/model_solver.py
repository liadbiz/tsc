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
import numpy as np
import pandas as pd

class Solver(object):
    def __init__(self, opt, model, dataset_name, num_classes):
        self.opt = opt
        self.model = model
        self.dataset_name = dataset_name
        self.num_classes = num_classes

    def fit(self, train_data, test_data,  optimizer, criterion, callbacks, metric):
        # train
        start_time = time.time()
        """ low level handling, not working for now, use build-in training process instead.
        for ep in range(self.opt.train.num_epochs):
            print("epoch {0} training start".format(ep))
            for step, (x, y) in enumerate(train_data):
                #print(y.shape)
                with tf.GradientTape() as tape:
                    logits = self.model(x)
                    loss = criterion(y, logits)
                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                #print(y, logits)
                train_metric(y, logits)
                if step % 5 == 0:
                    print('loss of step {0} is {1}'.format(step, loss.numpy()))
                    print('acc of step {0} is {1}'.format(step, train_metric.result().numpy()))
            print('training acc of epoch {0} is {1}'.format(ep, train_metric.result().numpy()))
        """
        # built-in training process
        #x_train, y_train = train_data
        self.model.compile(optimizer=optimizer, loss=criterion, metrics=[metric])
        #self.model.fit(x_train, y_train, batch_size=self.opt.train.batch_size, validation_split=0.2, epochs=self.opt.train.num_epochs)
        history = self.model.fit(train_data, validation_data=test_data, epochs=self.opt.train.num_epochs, callbacks=callbacks)

        end_time = time.time()
        duration = end_time - start_time
        print('duration of training dataset {0} is {1}'.format(self.dataset_name, duration))

        print('min validate accuracy: {0}'.format(np.max(history.history['val_categorical_accuracy'])))
        res = pd.DataFrame(history.history)
        res.to_csv()



    def save(self):
        pass

    def predict(self):
        pass

    def evaluate(self, test_data):
        # use best model in training process
        results = self.model.evaluate(test_data)
        print('test loss, test acc:', results)