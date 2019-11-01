#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Discourse Relation Sense Classifier

Feel free to change/restructure the code below
"""

__author__ = 'Jayeol Chun'


class DRSClassifier(object):
  """TODO: Implement a FeedForward Neural Network for Discourse Relation Sense
      Classification using Tensorflow/Keras (tensorflow 2.0)"""
  def __init__(self):
    self.build()

  def build(self):
    """TODO: Build your neural network here"""
    pass

  def train(self, train_instances, dev_instances):
    """TODO: Train the classifier on `train_instances` while evaluating
        periodically on `dev_instances`

    Args:
      train_instances: list
      dev_instances: list
    """
    pass

  def predict(self, instances, export_file="./preds.json"):
    """TODO: Given a trained model, make predictions on `instances` and export
        predictions to a json file

    Args:
      instances: list
      export_file: str, where to save your model's predictions on `instances`

    Returns:

    """
    pass
