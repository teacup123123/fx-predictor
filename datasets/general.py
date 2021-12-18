import os
import datetime
import pandas as pd
import requests
import csv


def download(code: str):
    got = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}", f"csv/{code}.csv")
    with open(f'csv/{code}.csv', 'wb') as f:
        f.write(got.content)


def parseCsv(code: str, topickle=True):
    got = []
    header = None
    with open(f'csv/{code}.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for date, value in spamreader:
            if header is None:
                header = date, value
                continue
            try:
                got.append((pd.to_datetime(date), float(value)))
            except:
                assert value == '.'
                # print('skipped unparseable data')
    df = pd.DataFrame(data=got, columns=header)
    if topickle: df.to_pickle(f'pickle/{code}.pickle')
    return df


def readpickle(code: str):
    return pd.read_pickle(f'pickle/{code}.pickle')


if __name__ == '__main__':
    for fn in os.listdir('py'):
        print(f'download/parse/pickling {fn}')
        fn = fn[:-3]  # 'without.py
        download(fn)
        parseCsv(fn, topickle=True)

    print('finished')
