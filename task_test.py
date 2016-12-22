#encoding: utf-8
import environment
import util

if __name__ == '__main__':
    prices = environment.get_prices('MSFT', '1992-07-22', '2016-07-22')
    util.plot_prices(prices)
