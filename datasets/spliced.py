import os.path
import pickle
import pandas as pd
import pickle as pk
import pylab as pl
import fred.grab_data as fred

# fred.DEXcodes
_dir, _ = os.path.split(__file__)


def sanitize_parse_fred(dexcode: str):
    t_rate_list = []
    renamed = fred.rename(dexcode)
    df: pd.DataFrame = pd.read_pickle(f'{_dir}/fred/pickle/{renamed}.pickle')
    a, b = renamed[-6:-3], renamed[-3:]

    for _, (t, rate) in df.iterrows():
        t_rate_list.append((t, rate if b == 'USD' else 1. / rate))
    return t_rate_list


if __name__ == '__main__':

    for c in fred.DEXcodes:
        t_rate_list = sanitize_parse_fred(c)
        ts, rates = zip(*t_rate_list)
        pl.figure()
        pl.title(f'{fred.rename(c)}(usd based)')
        pl.plot(ts, rates)
    pl.show()
