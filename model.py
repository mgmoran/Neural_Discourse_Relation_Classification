#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Discourse Relation Sense Classifier

Feel free to change/restructure the code below
"""

__author__ = 'Molly Moran'

import tensorflow as tf
import numpy as np
from preprocessing import LABELSET
import json

class DRSClassifier(object):
  def __init__(self,model_type,batches,epochs):
    self.model_type = model_type
    self.batches = batches
    self.epochs = epochs
    self.model = None
    self.build()

  def build(self):
      if self.model_type =='FFN':
          self.model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(30,300)),
                                                   tf.keras.layers.Dense(64, activation='relu'),  # first hidden layer
                                                   tf.keras.layers.Dense(32, activation='tanh'),  # second hidden layer
                                                   tf.keras.layers.Dense(21, activation='softmax')  # output layer
    ])
      elif self.model_type == 'CNN':
          self.model = tf.keras.models.Sequential([
              tf.keras.layers.Conv1D(filters=250,kernel_size=3,padding='valid',activation='relu', strides=1,input_shape=(30,300)),
              tf.keras.layers.GlobalMaxPooling1D(),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(150),
              tf.keras.layers.Dropout(0.2),
              tf.keras.layers.Dense(21,activation='softmax')])

      print(self.model.summary())
      self.model.compile(optimizer='adam',  # `adam` is a good initial choice when experimenting a new neural network
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

  def train(self, train_instances, dev_instances):
    train_features = np.asarray([rel['features'] for rel in train_instances])
    train_labels = np.asarray([rel['Sense'] for rel in train_instances])
    validation_features = np.asarray([rel['features'] for rel in dev_instances])
    validation_labels = np.asarray([rel['Sense'] for rel in dev_instances])
    self.model.fit(x=train_features,y=train_labels,batch_size=self.batches,epochs=self.epochs,validation_data=(validation_features,validation_labels))

  def predict(self, instances, export_file="./preds.json"):
    label_indices = {v: k for k, v in LABELSET.items()}
    processed = np.asarray([rel[0]['features'] for rel in instances])
    original = [pair[1] for pair in instances]
    preds = self.model.predict(processed)
    with open(export_file, 'w+') as f:
      for i in range(len(preds)):
        original_format = original[i]
        predicted_label = label_indices[np.argmax(preds[i])]
        Arg1 = [index[2] for index in original_format["Arg1"]["TokenList"]]
        Arg2 = [index[2] for index in original_format["Arg1"]["TokenList"]]
        Type = original_format["Type"]
        if Type == 'Implicit':
          Connective == []
        else:
          Connective = [index[2] for index in original_format["Connective"]["TokenList"]]
        DocID = original_format["DocID"]
        f.write(json.dumps({'Arg1': Arg1, 'Arg2': Arg2, 'Connective': Connective, 'DocID': DocID, 'Type': Type, 'Sense' : [predicted_label]}) + "\n")
    f.close()





