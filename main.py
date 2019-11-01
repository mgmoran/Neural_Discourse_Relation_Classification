#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Entry point for the Program Assignment 2

Feel free to change/restructure the code below
"""
from model import DRSClassifier
from preprocessing import load_data

__author__ = 'Jayeol Chun'


def run_scorer(preds_file):
  """Automatically runs `scorer.py` on model predictions

  TODO: You don't need to use this code if you'd rather run `scorer.py`
    manually.

  Args:
    preds_file: str, path to model's prediction file
  """
  import os
  import sys
  import subprocess

  if not os.path.exists(preds_file):
    print(
      "[!] Preds file `{}` doesn't exist in `run_scorer.py`".format(preds_file))
    sys.exit(-1)

  python = 'python3.5' # TODO: change this to your python command
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

def main():
  # loads and preprocesses data. See `preprocessing.py`
  data = load_data(data_dir='./data')

  # trains a classifier on `train` and `dev` set. See `model.py`
  clf = DRSClassifier()
  clf.train(train_instances=data['train'], dev_instances=data['dev'])

  # output model predictions on `test` set
  preds_file = "./preds.json"
  clf.predict(data['test'], export_file=preds_file)

  # measure the accuracy of model predictions using `scorer.py`
  run_scorer(preds_file)

if __name__ == '__main__':
  main()
