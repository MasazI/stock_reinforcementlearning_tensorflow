# encoding: utf-8
from yahoo_finance import Share
import numpy as np


def get_prices(share_symbol, start_date, end_date, cache_filename='cache_filename'):
    try:
        stock_prices = np.load(cache_filename)
    except IOError:
        share = Share(share_symbol)
        stock_hist = share.get_historical(start_date, end_date)
        stock_prices = [stock_price['Open'] for stock_price in stock_hist]
        np.save(cache_filename, stock_prices)
    return stock_prices


