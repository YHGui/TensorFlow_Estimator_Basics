# Tensorflow Estimator Basics

Train, predict, export and reload a `tf.estimator` for inference on a dummy example. This repository is forked and modified based on Guillaume Genthial's [repository](https://github.com/guillaumegenthial/tf-estimator-basics).

Please read the Guillaume Genthial's original [blog post](https://guillaumegenthial.github.io/serving-tensorflow-estimator.html) and my [blog post](https://leimao.github.io/blog/TensorFlow-Estimator-SavedModel/).

## Dependencies

* TensorFlow 1.14
* Python 3.6

## Quickstart

```
bash run.sh
```

## Details

- `model.py` defines the `model_fn`
- `train.py` trains an Estimator using the `model_fn`
- `export.py` exports the Estimator as a `saved_model`
- `predict.py` reloads an Estimator and uses it for prediction
- `fast_redict.py` reloads an Estimator and uses it for prediction using Marc Stogaitis's implementation.
- `faster_redict.py` reloads an Estimator and uses it for prediction using Lei Mao's implementation.
- `serve.py` reloads the inference graph from the `saved_model` format and uses it for prediction
