"""Predict using estimator.predict"""

__author__ = "Guillaume Genthial"


import functools
from pathlib import Path
import logging
import sys
import time

import tensorflow as tf

from model import model_fn


def example_input_fn(generator):
    """ An example input function to pass to predict. It must take a generator as input """

    def _inner_input_fn():
        dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32), output_shapes=(2)).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return {'feature': features}

    return _inner_input_fn


def my_service():
    """Some service yielding numbers"""
    start, end = 1, 10000
    for number in range(start, end):
        yield number


class TFEstimatorServe(object):

    def __init__(self, estimator, input_fn):
        self.data = []
        self.estimator = estimator
        self.input_fn = input_fn(self.data_generator)
        self.results = self.estimator.predict(input_fn=self.input_fn)
        self.closed = False
    
    def data_generator(self):

        while not self.closed:
            data = self.data.pop(0)
            yield data

    def predict(self, data):

        self.data = data
        predictions = []
        for _ in range(len(data)):
            predictions.append(next(self.results))
        return predictions

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")

if __name__ == '__main__':
    # Logging
    Path('model').mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('model/predict.log'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers

    # Instantiate estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='model',
                                       params={})
    
    server = TFEstimatorServe(estimator=estimator, input_fn=example_input_fn)

    server.predict(data=[[1,1]])

    # Predict using the estimator
    tic = time.time()
    for nb in my_service():
        pred = server.predict(data=[[nb, nb]])
        pred = pred[0]

        #print("------------------")
        #print((pred - 2*nb)**2)
        #print("------------------")
        

    toc = time.time()
    print('Average time in predict.py: {}s'.format((toc - tic) / 10))

