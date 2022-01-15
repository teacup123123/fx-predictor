import requests
from quick_query.xe import grab

import os

_dir, _ = os.path.split(__file__)

file = r'data_lionfx_swap.csv'  # manually downloaded from https://hirose-fx.co.jp/swap/lionfx_swap.csv


def update_swap_data():
    got = requests.get(r'https://hirose-fx.co.jp/swap/lionfx_swap.csv')
    string = got.content.decode('shift-jis')
    lines = string.splitlines()
    lines = [line.strip() for line in lines]
    header = lines[0]
    header = header.split(',')
    header[0] = 'date'
    for i in range(len(header) - 1, 0, -1):
        if i % 2 == 0:
            header[i] = header[i - 1]
    for i, (word, type) in enumerate(zip(header, lines[2].split(','))):
        if type.endswith('売り'):
            header[i] += '_sell'
        elif type.endswith('買い'):
            header[i] += '_buy'
    del lines[1:3]
    lines[0] = ','.join(header)
    with open('data_lionfx_swap.csv', 'w') as f:
        f.write('\n'.join(lines))


def update():
    rates = {}

    with open(rf'{_dir}/data_available_currencies.txt', 'r') as f:
        currencies = f.read(-1).split()
    for c in currencies:
        rate = grab(c, 'JPY')
        rates[c] = rate

    with open(rf'{_dir}/data_currency_now.txt', 'w') as f:
        f.write('\n'.join(f'{k} {v}' for k, v in rates.items()))


if __name__ == '__main__':
    update_swap_data()
