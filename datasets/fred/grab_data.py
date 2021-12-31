import re
import os
import datetime
import pandas as pd
import requests
import csv

_dir, _ = os.path.split(__file__)

from datasets.yahoo.currencies import currencies

toISO = {
    'CH': 'CNY',  # for RMB
    'UK': 'GBP',  # for GBP
    'AL': 'AUD',  # for AUD
    'SF': 'ZAR',  # for ZAR
    'SZ': 'CHF',  # for CHF
    'SD': 'SEK',  # for SEK
    'SI': 'SGD',  # for SGD
    'MA': 'MYR',  # for MYR
    'KO': 'KRW',  # for KRW
    'VZ': 'VEF',  # for VEF
    'BZ': 'BRL',  # for BRL
    'TH': 'THB',  # for THB
    'SL': 'LKR',  # for LKR
    'TA': 'NTD',  # for NTD
    'DN': 'DKK',  # for DKK
    'NO': 'NOK',  # for NOK
}
for _ in currencies:
    if _[:-1] not in toISO:
        toISO[_[:-1]] = _


def rename(dexcode: str):
    return f'rates_{toISO[dexcode[-2:]]}_{toISO[dexcode[-4:-2]]}'


def download(dexcode: str):
    got = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={dexcode}")
    with open(f'{_dir}/csv_data/{rename(dexcode)}.csv', 'wb') as f:
        f.write(got.content)


def parseCsv(renamed: str, topickle=True):
    got = []
    header = None
    with open(f'{_dir}/csv_data/{renamed}.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for date, value in spamreader:
            if header is None:
                header = date, value
                continue
            try:
                got.append((pd.to_datetime(date + ' 12:00 -5', utc=True), float(value)))
            except:
                assert value == '.'
    df = pd.DataFrame(data=got, columns=header)
    if topickle: df.to_pickle(f'{_dir}/pickle/{renamed}.pickle')
    return df


def readpickle(renamed: str):
    return pd.read_pickle(f'{_dir}/pickle/{renamed}.pickle')


def grab_codes():
    got = requests.get('https://fred.stlouisfed.org/categories/94')
    content = got.content.decode('utf-8')
    pattern = '''<a href="/series/DEX(..)(..)" style="font-size:1\.2em" class="series-title pager-series-title-gtm">'''
    pairs = re.findall(pattern, content)

    pair_names = []
    for a, b in pairs:
        pair_names.append(f'DEX{a}{b}')
    return pair_names


DEXcodes = grab_codes()

if __name__ == '__main__':
    print()
    for dexcode in DEXcodes:
        download(dexcode)
        renamed = rename(dexcode)
        print(f'download/parse/pickling {renamed}')
        parseCsv(renamed, topickle=True)

    print('finished')
