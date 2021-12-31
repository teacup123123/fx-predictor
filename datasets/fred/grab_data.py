import re
import os
import datetime
import pandas as pd
import requests
import csv

_dir, _ = os.path.split(__file__)


def download(code: str):
    got = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}")
    with open(f'{_dir}/csv_data/{code}.csv', 'wb') as f:
        f.write(got.content)


def parseCsv(code: str, topickle=True):
    got = []
    header = None
    with open(f'{_dir}/csv_data/{code}.csv', newline='\n') as csvfile:
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
    if topickle: df.to_pickle(f'{_dir}/pickle/{code}.pickle')
    return df


def readpickle(code: str):
    return pd.read_pickle(f'{_dir}/pickle/{code}.pickle')


def grab_codes():
    got = requests.get('https://fred.stlouisfed.org/categories/94')
    content = got.content.decode('utf-8')
    pattern = '''<a href="/series/DEX(..)(..)" style="font-size:1\.2em" class="series-title pager-series-title-gtm">'''
    pairs = re.findall(pattern, content)
    codes = set()
    toISO = {}
    from datasets.yahoo.currencies import currencies

    def correspondance(x):
        for c in currencies:
            if x in c:
                return (x, c)
        return (None, None)

    for a, b in pairs:
        x, y = correspondance(a)
        if x: codes.add(x);toISO[x] = y
        x, y = correspondance(b)
        if x: codes.add(x);toISO[x] = y

    DEXcodes = []
    for a, b in pairs:
        if a in codes and b in codes:
            DEXcodes.append(f'DEX{a}{b}')
    return toISO, DEXcodes


toISO, DEXcodes = grab_codes()

if __name__ == '__main__':
    print()
    for fn in DEXcodes:
        print(f'download/parse/pickling {fn}')
        download(fn)
        parseCsv(fn, topickle=True)

    print('finished')
