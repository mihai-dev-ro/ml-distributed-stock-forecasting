# Importing the libraries
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from os import path


def get_symbol_csv_filepath(symbol, mode='daily'):
    if (mode == 'intraday'):
        return f'../data/{symbol}_intraday_dataset.csv'
    else:
        return f'../data/{symbol}_dataset.csv'

def save_symbol_dataset(symbol, mode='daily'):
    ts = TimeSeries(key='02AF0F8GA45L20YP', output_format='pandas')
    
    if (mode == 'intraday'):
        data, meta_data = ts.get_intraday(symbol, outputsize='full')
    else:
        data, meta_data = ts.get_daily(symbol, outputsize='full')
        
    data.to_csv(get_symbol_csv_filepath(symbol, mode))
    
def get_symbol_dataset(symbol, mode='daily'):
    # if data is not available, download it and serve it afterwards
    csv_path = get_symbol_csv_filepath(symbol, mode)
    if not path.exists(csv_path):
        save_symbol_dataset(symbol, mode)
    data = pd.read_csv(csv_path, parse_dates=['date'])
    return data
  
def load_data(lst_symbols):
  dct_data = {}
  for symbol in lst_symbols:
    dct_data[symbol] = get_symbol_dataset(symbol)
  return dct_data

