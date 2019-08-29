
import functools
from pathlib import Path
import logging
import sys
import time

import tensorflow as tf

from model import model_fn


class FastPredict(object):

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature) 
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(
                input_fn=self.input_fn(self._create_generator))
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(feature_batch)))

        results = []
        for _ in range(self.batch_size):
            results.append(next(self.predictions))
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")


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
    start, end = 1, 100
    for number in range(start, end):
        yield number

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

    predictor = FastPredict(estimator=estimator, input_fn=example_input_fn)

    predictor.predict([1,1])

    # Predict using the estimator
    count = 0
    tic = time.time()
    for nb in my_service():
        count += 1
        pred = predictor.predict([nb, nb])
        pred = pred[0]

        # print((pred - 2*nb)**2)

    toc = time.time()
    print('Average time in predict.py: {}s'.format((toc - tic) / count))