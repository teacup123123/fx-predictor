import os
from typing import List

import pandas as pd
import pylab as pl

from datasets.t_rate_list_format import TimeSeriesSingle

_dir, _ = os.path.split(__file__)

import datasets.fred.grab_data as gd


def sanitize_parse_fred(dexcode: str):
    t_rate_list = TimeSeriesSingle()
    renamed = gd.rename(dexcode)
    df: pd.DataFrame = pd.read_pickle(f'{_dir}/pickle/{renamed}.pickle')
    a, b = renamed[-6:-3], renamed[-3:]

    for _, (t, rate) in df.iterrows():
        t_rate_list.append((t, rate if b == 'USD' else 1. / rate))
    return t_rate_list


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


def initialize(list_of_codes: List[str] = None):
    if list_of_codes is None: list_of_codes = []
    for i, x in enumerate(list_of_codes): list_of_codes[i] = list_of_codes[i].upper()
    for c in gd.DEXcodes:
        currency = c[3:].replace('US', '')
        if gd.toISO[currency] not in list_of_codes: continue
        print(f'inititializing {currency} FRED data')
        t_rate_list = sanitize_parse_fred(c)
        setattr(as_TSS, gd.toISO[currency], t_rate_list)

    # small example
    # as_TSS.JPY.inverse.plot()
    # pl.show()


if __name__ == '__main__':
    initialize(['jpy'])
