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
from datasets.t_rate_list_format import TimeSeriesMerged, TimeSeriesSingle

core = 'EUR CHF JPY NZD gbp usd CAD'
rim = 'sek mxn aud'
correlation = f'{core} {rim}'.upper().split()

if __name__ == '__main__':

    # load fred
    # if True:
    #     fred.initialize(correlation)
    #     with open('initialized.pickle', 'wb') as f:
    #         pk.dump(fred.as_TSS.as_dict(), f)

    # fred -> merged logged
    DAYS = 365 * 20
    if True:
        period = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=DAYS), pd.Timestamp.now(tz='UTC'))
        with open('initialized.pickle', 'rb') as f:
            fred.as_TSS.load_dict(pk.load(f))
        fake_usd = TimeSeriesSingle()
        for _ in period: fake_usd.append((_, 1.))
        _ = fred.as_TSS.as_dict()
        _['USD'] = fake_usd
        merged = TimeSeriesMerged(period, _)
        mergedlogged = merged.log100
        with open('mergedlogged.pickle', 'wb') as f:
            pk.dump(mergedlogged, f)

    with open('mergedlogged.pickle', 'rb') as f:
        mergedlogged: TimeSeriesMerged = pk.load(f)
        geom = 0.
        for curve in mergedlogged.as_np:
            geom += curve
        geom /= len(mergedlogged.currencies)

        for currency in mergedlogged.currencies:
            mergedlogged[currency] = mergedlogged[currency] - geom

    high_passed: TimeSeriesMerged = mergedlogged.copy()
    for currency, curve in zip(high_passed.currencies, high_passed.as_np):
        comeback = np.concatenate((curve, np.flip(curve, axis=0)))
        freqs = np.fft.fftfreq(len(comeback), d=1)
        filter = np.abs(freqs)
        comeback = np.fft.ifft(np.fft.fft(comeback) * filter)
        curve = comeback[:len(high_passed.merged_times)]
        high_passed[currency] = curve.real

    covar = np.cov(high_passed.as_np)
    ev, P = np.linalg.eigh(covar)
    P = P * 2. / np.sum((np.abs(P)), axis=0, keepdims=True)
    ev = np.diag(P.T @ covar @ P)

    test = P @ np.diag(ev) @ P.T

    print(np.ones((1, len(mergedlogged.currencies))) @ P)

    MAX_EV = 6  # len(mergedlogged.currencies)
    P = np.flip(P, axis=1)
    ev = np.diag(P.T @ covar @ P)
    sev = np.sqrt(np.abs(ev))

    volitility = pd.DataFrame(P[:, :MAX_EV], index=mergedlogged.currencies,
                              columns=[f'SE{f:.2e}' for f in sev[:MAX_EV]])
    print(volitility)

    pl.figure()
    pl.title('mergedlogged')
    for i, c in enumerate(P.T @ mergedlogged.as_np):
        # if i == 0: continue
        if i == MAX_EV: break
        pl.plot(mergedlogged.merged_times, np.abs(c))
    pl.legend([f'E{x:.2e}' for x in sev][:MAX_EV])

    pl.figure()
    pl.title('lowcut')
    for i, c in enumerate(P.T @ (mergedlogged.as_np - high_passed.as_np)):
        # if i == 0: continue
        if i == MAX_EV: break
        pl.plot(mergedlogged.merged_times, np.abs(c))
    pl.legend([f'E{x:.2e}' for x in sev][:MAX_EV])

    pl.figure()
    pl.title('high_passed')
    for i, c in enumerate(P.T @ high_passed.as_np):
        # if i == 0: continue
        if i == MAX_EV: break
        pl.plot(high_passed.merged_times, np.abs(c))
    pl.legend([f'E{x:.2e}' for x in sev][:MAX_EV])
    pl.show()
    # fred.as_TSS.JPY.plot()
    # pl.show()
    print()
