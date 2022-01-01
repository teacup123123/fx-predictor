'''
A,B,C are countries/currencies
the absolute strength of each currency X:
log X = internal_affairs(X) + connectivity(X~Y, for Y trade partners)

We can only see log(X)-log(Y) as the log of FX rates
can we deduce internal_affairs(X) and connectivity(X~Y)?

the best we can do might be to get

internal_affairs(X) - mean[internal_affairs(Y), all Y]
To simplify,
each country X wants some amount of Y for Y connected to X, the relative
the want of Y by X will drive up the price of Y, and down the price of X

Internal affair
A B C
↑ x x

Export of A to B = investment by B into A:
A B C
↑ ↓ x

The effect will be stronger the smaller the economic volume of the country

PCA on the log'd currency pairs should yield insight on the different vars,
The relevant timescale should be 10years handwavingly
'''
import numpy as np
import pandas as pd
import pickle as pk

import pylab as pl

import datasets.fred.unified_format as fred
from datasets.t_rate_list_format import TimeSeriesMerged

core = 'CHF JPY NZD gbp usd CAD'
rim = 'sek mxn aud'
correlation = f'{core} {rim}'.upper().split()

if __name__ == '__main__':
    DAYS = 365 * 15

    # load fred
    if False:
        fred.initialize(correlation)
        with open('initialized.pickle', 'wb') as f:
            pk.dump(fred.as_TSS.as_dict(), f)

    # fred -> merged logged
    if False:
        period = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=DAYS), pd.Timestamp.now(tz='UTC'))
        with open('initialized.pickle', 'rb') as f:
            fred.as_TSS.load_dict(pk.load(f))
        merged = TimeSeriesMerged(period, fred.as_TSS.as_dict())
        mergedlogged = merged.log100.remove_trend
        with open('mergedlogged.pickle', 'wb') as f:
            pk.dump(mergedlogged, f)

    with open('mergedlogged.pickle', 'rb') as f:
        mergedlogged: TimeSeriesMerged = pk.load(f)

    high_passed: TimeSeriesMerged = mergedlogged.copy()
    for currency, curve in zip(high_passed.currencies, high_passed.as_np):
        comeback = np.concatenate((curve, np.flip(curve, axis=0)))
        freqs = np.fft.fftfreq(len(comeback), d=1)
        filter = np.abs(freqs)
        comeback = np.fft.ifft(np.fft.fft(comeback) * filter)
        curve = comeback[:len(high_passed.merged_times)]
        high_passed[currency] = curve.real

    covar = np.cov(high_passed.as_np)
    ev, P = np.linalg.eig(covar)
    test = P @ np.diag(ev) @ P.T
    volitility = pd.DataFrame(P, index=mergedlogged.currencies, columns=[f'E{f:.2e}' for f in ev])
    print(volitility)

    pl.figure()
    pl.title('mergedlogged')
    for c in P.T @ mergedlogged.as_np:
        pl.plot(c)
    pl.legend([f'E{x:.2e}' for x in ev])

    pl.figure()
    pl.title('lowcut')
    for c in P.T @ (mergedlogged.as_np - high_passed.as_np):
        pl.plot(c)
    pl.legend([f'E{x:.2e}' for x in ev])

    pl.figure()
    pl.title('high_passed')
    for c in P.T @ high_passed.as_np:
        pl.plot(c)
    pl.legend([f'E{x:.2e}' for x in ev])
    pl.show()
    # fred.as_TSS.JPY.plot()
    # pl.show()
    print()
