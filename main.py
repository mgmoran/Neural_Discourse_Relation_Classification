#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Molly Moran
Stat NLP, PA2"""
from model import DRSClassifier
from preprocessing import load_data

### A configurable Neural Network trained on tagged data from the Penn Discourse
### Treebank, motivated by the CoNLL 2016 Shared Task: Shallow Discourse Parsing.

##Starter code - returns P/R/F stats for explicit and implicit relations
def run_scorer(preds_file):
  import os
  import sys
  import subprocess

  if not os.path.exists(preds_file):
    print(
      "[!] Preds file `{}` doesn't exist in `run_scorer.py`".format(preds_file))
    sys.exit(-1)

  python = 'python3.7' # TODO: change this to your python command
  scorer = './scorer.py'
  gold = './data/test/relations.json'
  auto = preds_file
  command = "{} {} {} {}".format(python, scorer, gold, auto)

  print("Running scorer with command:", command)
  proc = subprocess.Popen(
    command, stdout=sys.stdout, stderr=sys.stderr, shell=True,
    universal_newlines=True
  )
  proc.wait()

### Running the model ###
def main():
  data = load_data(data_dir='./data')
  ### Edit any hyperparameters here, including model type. ###
  clf = DRSClassifier(model_type='FFN',batches=64,epochs=5)
  clf.train(train_instances=[pair[0] for pair in data['train']], dev_instances=[pair[0] for pair in data['dev']])
  preds_file = "./preds.json"
  clf.predict(data['test'], export_file=preds_file)

  run_scorer(preds_file)

if __name__ == '__main__':
  main()

