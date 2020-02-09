from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys as Modes

from sklearn import preprocessing
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input
from keras import optimizers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

SEQUENCE_INPUT = 55
SEQUENCE_OUTPUT = 5
SEQUENCE_LENGTH = SEQUENCE_INPUT + SEQUENCE_OUTPUT

def input_fn(filename, batch_size=100, shuffle=False):
  # read csv file into pandas dataframe (2-dimensional, size-mutable data)
  df = pd.read_csv(filename, parse_dates=['date'])
  
  # drop the date column
  input_cleansed = df.drop('date', axis=1)
  
  # normalize the data
  input_normalizer = preprocessing.MinMaxScaler()
  input_normalized = input_normalizer.fit_transform(input_cleansed)
  
  y_normalizer = preprocessing.MinMaxScaler()
  y_values = input_cleansed[:,0].copy()
  y_normalizer.fit(y_values)
  y_normalized = y_normalizer.transform(y_values)
  
  # create the M window datasets
  np_features = np.array([input_normalized[i:i+SEQUENCE_INPUT].copy() 
                     for i in range(len(input_normalized) - SEQUENCE_LENGTH)])
  np_targets = np.array([y_normalized[i:i+SEQUENCE_OUTPUT]
                          for i in range(SEQUENCE_INPUT, len(y_normalized) - 
                                         SEQUENCE_OUTPUT)])
  
  
  dataset = tf.data.Dataset.from_tensor_slices((np_features, np_targets))
  if shuffle:
    dataset = dataset.shuffle()
  
  dataset = dataset.batch(batch_size)
  dataset.repeat()
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

def get_input_fn(filename, batch_size=100, shuffle=False):
  return lambda: input_fn(filename, batch_size, shuffle)


def get_rnn_model():
  
  # input layer
  tf_input = Input(shape=(SEQUENCE_INPUT, 5), name='Net_input')
  
  #lstm
  lyr_lstm1 = LSTM(150, return_sequences = True, name='Net_lstm1')(tf_input)
  
  #lstm
  lyr_lstm2 = LSTM(150, name='Net_lstm2')(lyr_lstm1)
  lyr_dropout1 = Dropout(rate=0.5, name='Net_lstm2_dropout')(lyr_lstm2)
  
  # dense1
  lyr_dense1 = Dense(128, activation='tanh', name='Net_dense1')(lyr_dropout1)
  lyr_dropout2 = Dropout(rate=0.5, name='Net_lstm2_dropout')(lyr_dense1)
  
  
  # dense2
  tf_predictions = Dense(SEQUENCE_OUTPUT, activation='relu', name='Net_dense2')(lyr_dropout2)

  regressor = Model(inputs=tf_input, outputs=tf_predictions, 
                    name='Regressor_stock_prices')
  
  # metrics
  #def model_metrics_mse(y_true, y_pred):
    
  
  adam = optimizers.Adam(learning_rate=0.0009)
  regressor.compile(optimizer=adam, loss='mse') 
  regressor.summary()
  
  # build the estimator
  return regressor


def build_estimator(model_dir):
  return tf.keras.estimator.model_to_estimator(
    keras_model=get_rnn_model(), 
    model_dir,
    config=tf.estimator.RunConfig(save_checkpoints_secs=180))


# def serving_input_fn_0():
#   inputs = {'inputs': tf.compat.v1.placeholder(tf.float32, [None, 784])}
#   return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def get_serving_input_fn(train_data_file_name):
  
  # normalize the data with the same normalizer
  # read csv file into pandas dataframe (2-dimensional, size-mutable data)
  df_train = pd.read_csv(train_data_file_name, parse_dates=['date'])
  
   # drop the date column
  input_cleansed = df_train.drop('date', axis=1)
  
  # normalize the data
  input_normalizer = preprocessing.MinMaxScaler()
  input_normalizer.fit(input_cleansed)
  
  # normalize the inputs
  
  inputs = tf.compat.v1.placeholder(tf.float32, [SEQUENCE_INPUT,5],
                                    name='stock_data_history')
  features = input_normalizer.transform(inputs)
  
  
  return tf.estimator.export.ServingInputReceiver(features, inputs)
  

def train_and_evaluate(output_dir,
                       data_dir,
                       train_batch_size=100,
                       eval_batch_size=100,
                       train_steps=10000,
                       eval_steps=100,
                       **experiment_args):
  
  estimator = build_estimator(output_dir)
  
  train_spec=tf.estimator.TrainSpec(
          input_fn=get_input_fn(
              filename=os.path.join(data_dir, 'MSFT_train_dataset.csv'),
              batch_size=train_batch_size,
              shuffle=True),
          max_steps=train_steps)
  
  exporter = tf.estimator.LatestExporter(
    'exporter', 
    serving_input_receiver_fn=get_serving_input_fn())
  
  eval_spec=tf.estimator.EvalSpec(
          input_fn=get_input_fn(
              filename=os.path.join(data_dir, 'MSFT_test_dataset.csv'),
              batch_size=eval_batch_size,
              shuffle=False),
          steps=eval_steps,
          exporters=exporter)
  
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec,
                                  **experiment_args)
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  