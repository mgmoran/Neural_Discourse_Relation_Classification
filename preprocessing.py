#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Molly Moran
Stat NLP, PA2"""

import codecs
import json
import os
from nltk.tokenize import RegexpTokenizer
from pymagnitude import Magnitude
import numpy as np
from collections import defaultdict

__author__ = 'Molly Moran'
"""Useful constants when processing `relations.json`"""
ARG1  = 'Arg1'
ARG2  = 'Arg2'
CONN  = 'Connective'
SENSE = 'Sense'
TYPE  = 'Type'
KEYS  = [ARG1, ARG2, CONN, SENSE]
TEXT  = 'RawText'
EMBEDDINGS =  Magnitude("wiki-news-300d-1M-subword.magnitude", case_insensitive=True,normalized=False)
VOCAB = []
LABELSET = defaultdict(int)
tokenizer = RegexpTokenizer(r'[\'\w\-]+|[.?!;:,]')
UNK = None

def featurize(rel,argtrunc=10,conntrunc=10):
  rel_rep = {"Type": rel[TYPE]}

  ## explicit connectives ##
  if rel[CONN] != []:
      rel[CONN] = rel[CONN][:conntrunc] ## truncating connective to 5 tokens ##
      diff = conntrunc - len(rel[CONN]) ## calculating difference if connective is shorter than 5 tokens ##
      CONN_vector = np.pad([EMBEDDINGS.query(token)
                        if token.lower() in VOCAB else UNK ## handling unknown words with an averaged word embedding
                        for token in rel[CONN]], [(0, diff), (0, 0)], mode='constant') ## padding if needed

  ## implicit connectives get arrays of 0s
  else:
    CONN_vector = np.zeros((conntrunc,300))

  ## truncating arguments ##
  arg1 = rel[ARG1][:argtrunc]
  arg2 = rel[ARG2][:argtrunc]

  ## how much padding is needed ##
  padding1 = argtrunc - len(arg1)
  padding2 = argtrunc - len(arg2)

  ## padding (if needed) and handling unkonwns with an averaged word embedding
  arg1_vector = np.pad([EMBEDDINGS.query(token)
                        if token.lower() in VOCAB else UNK
                        for token in arg1], [(0, padding1), (0, 0)], mode='constant')
  arg2_vector = np.pad([EMBEDDINGS.query(token)
                        if token.lower() in VOCAB else UNK
                        for token in arg2], [(0, padding2), (0, 0)], mode='constant')

  rel_vector1 = np.concatenate((CONN_vector,arg1_vector))

  ## concatenating arg1, connective, and arg2 together (in that order) ##
  rel_vector = np.concatenate((rel_vector1,arg2_vector))
  rel_rep['features'] = rel_vector

  ##vectorizing label ##
  sense = rel[SENSE]
  sensevector = np.zeros((len(LABELSET)))
  senseindex = LABELSET[sense]
  sensevector[senseindex] = 1.0
  rel_rep[SENSE] = sensevector

  return rel_rep

def preprocess(rel):
  # Preprocessing: I do not filter out punctuation or stopwords,
  # since I think those could be crucial in predicting discourse relations.
  # I use the nltk RegExp tokenizer to separate punctuation from words, except when it's a word-internal hyphen or apostrophe.
  # I use magnitude word vectors, which are case-insensitive, so that takes care of capitalization normalization.
  rel_dict = {}
  for key in KEYS:
    if key in [ARG1, ARG2, CONN]:
      # for `Arg1`, `Arg2`, `Connective`, we only keep tokens of `RawText`
      rawtext = rel[key][TEXT]
      tokenized = tokenizer.tokenize(rawtext)
      rel_dict[key] = tokenized
    elif key == SENSE:
      # `Sense` is the target label. For relations with multiple senses, we
      # assume (for simplicity) that the first label is the gold standard.
      rel_dict[key] = rel[key][0]
  rel_dict[TYPE] = rel[TYPE]

  ## making sure that implicit annotations are not used ##
  if rel[TYPE] == 'Implicit':
    rel[CONN] = []

  # featurizing them
  feat_rel = featurize(rel_dict)
  return feat_rel

def extract_vocab(train,dev):
  global UNK
  global LABELSET
  global VOCAB
  i = 0
  training_vocab = []
  for path in (train, dev):
    rel_path = os.path.join(path, "relations.json")
    assert os.path.exists(rel_path), \
    "{} does not exist in `load_relations.py".format(rel_path)

  # grabbing vocab and labels from train and dev sets #
    with codecs.open(rel_path, encoding='utf-8') as pdtb:
      for pdtb_line in pdtb:
        rel = json.loads(pdtb_line)
        for key in [ARG1, ARG2, CONN]:
          training_vocab.extend((tokenizer.tokenize(rel[key][TEXT])))
        if rel[SENSE][0] not in LABELSET:
          LABELSET[rel[SENSE][0]] = i
          i += 1
  VOCAB = set([token.lower() for token in training_vocab])
  UNK = np.transpose(sum([EMBEDDINGS.query(token) for token in VOCAB]) / len(VOCAB))

def load_relations(data_file):
  """Loads a single `relations.json` file

  Args:
    data_file: str, path to a single data file

  Returns:
    list, where each item is of type dict
  """
  processed = []
  original =[]
  rel_path = os.path.join(data_file, "relations.json")
  with codecs.open(rel_path, encoding='utf-8') as pdtb:
    for pdtb_line in pdtb:
      rel = json.loads(pdtb_line)
      original.append(rel)
      rel = preprocess(rel)
      processed.append(rel)

  ### rels is the featurized docs(relations)
  return list(zip(processed,original))

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
  train_path = os.path.join(data_dir,"train")
  dev_path = os.path.join(data_dir,"dev")
  extract_vocab(train_path, dev_path)
  for folder in os.listdir(data_dir):
    print("Loading", folder)
    folder_path = os.path.join(data_dir, folder)
    data[folder] = load_relations(folder_path)
  return data
