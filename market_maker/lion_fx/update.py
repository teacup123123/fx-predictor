from quick_query.xe_com import grab

import os

_dir, _ = os.path.split(__file__)

file = r'data_lionfx_swap.csv'  # manually downloaded from https://hirose-fx.co.jp/swap/lionfx_swap.csv

def update():
    rates = {}

    with open(rf'{_dir}/data_available_currencies.txt', 'r') as f:
        currencies = f.read(-1).split()
    for c in currencies:
        rate = grab(c, 'JPY')
        rates[c] = rate

    with open(rf'{_dir}/data_currency_now.txt', 'w') as f:
        f.write('\n'.join(f'{k} {v}' for k, v in rates.items()))

def sanitize_csv():
    with open(file, 'r') as f:
        lines = f.readlines()

    _cache = '?'

    # def format(i, token: str):
    #     global _cache
    #     if i == 0:
    #         return token
    #     else:
    #         if i % 2 == 1:
    #             _cache = token
    #             return _cache + '_sell' if not _cache.endswith('_sell') else _cache
    #         else:
    #             return _cache + '_buy' if not _cache.endswith('_buy') else _cache

    # lines[0] = ','.join([format(i, nature) for i, nature in enumerate(lines[0].split(','))]) + '\n'
    #
    # with open(file, 'w') as f:
    #     f.writelines(lines)