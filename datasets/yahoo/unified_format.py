import re
import os
from collections import defaultdict
from typing import List
import pickle as pk

import pandas as pd
import pylab as pl

from datasets.t_rate_list_format import TimeSeriesSingle

_dir, _ = os.path.split(__file__)


def init_pickles(per_str, resolution_str):
    chunks = []
    currency_2_curve = defaultdict(TimeSeriesSingle)
    for file in os.listdir(r'pickles'):
        attempt_re = re.match(f'accu_{per_str}_{resolution_str}_([0-9]+)_([0-9]+).pickle', file)
        if attempt_re:
            start, end = attempt_re.groups()
            print(f'{start}->{end}')
            with open(os.path.join('pickles', file), 'rb') as f:
                ts, full_data = pk.load(f)
            full_data: dict
            for currency, curve in full_data.items():
                for t, c in zip(ts, curve):
                    currency_2_curve[currency].append((pd.Timestamp(t, tz='GMT', unit='s'), c))
    for currency, curve in currency_2_curve.items():
        curve.sort()
        setattr(as_TSS, currency, curve)


class as_TSS:
    CHF = TimeSeriesSingle()
    JPY = TimeSeriesSingle()
    SGD = TimeSeriesSingle()
    NZD = TimeSeriesSingle()
    GBP = TimeSeriesSingle()
    HKD = TimeSeriesSingle()
    PLN = TimeSeriesSingle()
    HUF = TimeSeriesSingle()
    ZAR = TimeSeriesSingle()
    EUR = TimeSeriesSingle()
    CAD = TimeSeriesSingle()
    TRY = TimeSeriesSingle()
    SEK = TimeSeriesSingle()
    AUD = TimeSeriesSingle()
    NOK = TimeSeriesSingle()
    USD = TimeSeriesSingle()
    MXN = TimeSeriesSingle()

    @staticmethod
    def is_legal_inited(currency, v):
        if currency.startswith('_'): return False
        if isinstance(v, staticmethod): return False
        return len(v) > 0

    @staticmethod
    def as_dict():
        result = {k: v for k, v in as_TSS.__dict__.items() if as_TSS.is_legal_inited(k, v)}
        return result

    @staticmethod
    def load_dict(d: dict):
        for k, v in d.items():
            if as_TSS.is_legal_inited(k, v):
                setattr(as_TSS, k, v)


if __name__ == '__main__':
    init_pickles('2y', '60m')
    # as_TSS.USD.plot()
    as_TSS.JPY.plot()
    pl.show()
