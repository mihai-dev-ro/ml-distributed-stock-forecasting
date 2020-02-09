# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from trainer import data_loader

FLAGS = None

def get_symbol_csv_filepath(symbol, mode='train'):
    if (mode == 'train'):
        return f'../data/{symbol}_train_dataset.csv'
    else:
        return f'../data/{symbol}_test_dataset.csv'
      
def save_symbol_dataset(symbol):
    ts = TimeSeries(key='02AF0F8GA45L20YP', output_format='pandas')
    
    data, meta_data = ts.get_daily(symbol, outputsize='full')
    n_split = int(len(data) * 0.9)
    data[:n_split].to_csv(get_symbol_csv_filepath(symbol, 'train'))
    data[n_split:].to_csv(get_symbol_csv_filepath(symbol, 'test'))
    
def ensure_symbol_dataset(symbol):
    # if data is not available, download it and serve it afterwards
    csv_path = get_symbol_csv_filepath(symbol, 'train')
    if not path.exists(csv_path):
        save_symbol_dataset(symbol)

def main(unused_argv):
  #lst_stocks = ['TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NFLX', 'FB']
  lst_stock_symbols = [FLAGS.symbol]
  ensure_symbol_dataset(lst_stock_symbols)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--symbol',
      type=str,
      default='MSFT',
      help='Symbol for which to load the stock market historical data'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
