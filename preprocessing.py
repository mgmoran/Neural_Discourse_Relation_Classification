#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Data Loader/Pre-processor Functions

Feel free to change/restructure the code below
"""
import codecs
import json
import os

__author__ = 'Jayeol Chun'


"""Useful constants when processing `relations.json`"""
ARG1  = 'Arg1'
ARG2  = 'Arg2'
CONN  = 'Connective'
SENSE = 'Sense'
TYPE  = 'Type'
KEYS  = [ARG1, ARG2, CONN, SENSE]
TEXT  = 'RawText'

def featurize(rel):
  """Featurizes a relation dict into feature vector

  TODO: `rel` is a dict object for a single relation in `relations.json`, where
    `Arg1`, `Arg2`, `Connective` and `Sense` are all strings (`Conn` may be an
    empty string). Implement a featurization function that transforms this into
    a feature vector. You may use word embeddings.

  Args:
    rel: dict, see `preprocess` below

  Returns:

  """
  return rel

def preprocess(rel):
  """Preprocesses a single relation in `relations.json`

  Args:
    rel: dict

  Returns:
    see `featurize` above
  """
  rel_dict = {}
  for key in KEYS:

    if key in [ARG1, ARG2, CONN]:
      # for `Arg1`, `Arg2`, `Connective`, we only keep tokens of `RawText`
      rel_dict[key] = rel[key][TEXT]

    elif key == SENSE:
      # `Sense` is the target label. For relations with multiple senses, we
      # assume (for simplicity) that the first label is the gold standard.
      rel_dict[key] = rel[key][0]

  # into feature vector/matrix
  feat_rel = featurize(rel_dict)

  return feat_rel

def load_relations(data_file):
  """Loads a single `relations.json` file

  Args:
    data_file: str, path to a single data file

  Returns:
    list, where each item is of type dict
  """
  rel_path = os.path.join(data_file, "relations.json")
  assert os.path.exists(rel_path), \
    "{} does not exist in `load_relations.py".format(rel_path)

  rels = []
  with codecs.open(rel_path, encoding='utf-8') as pdtb:
    for pdtb_line in pdtb:
      rel = json.loads(pdtb_line)
      rel = preprocess(rel)
      rels.append(rel)

  return rels

def load_data(data_dir='./data'):
  """Loads all data in `data_dir` as a dict

  Each of `dev`, `train` and `test` contains (1) `raw` folder (2)
    `relations.json`. We don't need to worry about `raw` folder, and instead
    focus on `relations.json` which contains all the information we need for our
    classification task.

  Args:
    data_dir: str, the root directory of all data

  Returns:
    dict, where the keys are: `dev`, `train` and `test` and the values are lists
      of relations data in `relations.json`
  """
  assert os.path.exists(data_dir), "`data_dir` does not exist in `load_data`"

  data = {}
  for folder in os.listdir(data_dir):
    print("Loading", folder)
    folder_path = os.path.join(data_dir, folder)
    data[folder] = load_relations(folder_path)

  return data
