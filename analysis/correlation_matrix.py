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
import itertools

import numpy as np
import pandas as pd
import pickle as pk

import pylab as pl

from datasets.t_rate_list_format import TimeSeriesMerged, TimeSeriesSingle

core = 'EUR CHF JPY NZD gbp usd CAD'
rim = 'mxn aud'
# dataset = 'fred'
dataset = 'yahoo'
correlation = f'{core} {rim}'.upper().split()
parameters = {
    'init': False,
    'log100_removetrend': False
}


def high_passed(mergedlogged):
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
    return high_passed


def svd(high_passed):
    covar = np.cov(high_passed.as_np)
    _, P = np.linalg.eigh(covar)
    order = np.argsort(_)
    order = np.flip(order)
    P = P[:, order]

    costeffectiveness = 2. / np.sum((np.abs(P)), axis=0, keepdims=True)
    P = P * costeffectiveness

    Pinv = np.linalg.inv(P)
    ev = np.diag(Pinv @ covar @ P)

    # test = P @ np.diag(ev) @ Pinv
    return ev, P, Pinv, covar


def gridify(c, profit=0.5):  # 5 euros is nice...
    tmask = np.zeros(c.size, dtype=bool)
    gridmid = c[0]
    gridified = []
    ok = False
    for i, x in enumerate(c):
        if gridmid - profit / 2 > x:
            ok, gridmid = True, x + profit / 2
        elif gridmid + profit / 2 < x:
            ok, gridmid = True, x - profit / 2
        if ok:
            gridified.append(gridmid)
        tmask[i] = ok
    return tmask, np.array(gridified)


def derivate(c):
    center = np.mean(c)
    cross = [i for i, _ in enumerate(c[:-1]) if (c[i] - center) * (c[i + 1] - center) < 0]
    B = np.roll(c, -1) - np.roll(c, +1)
    tmask = np.zeros(c.size, dtype=bool)
    tmask[cross[0]:cross[-1]] = True
    tmask[0] = False
    tmask[-1] = False
    return tmask, B[tmask]


def profitability(msk1, goal, msk2, benchmarked, times_length):
    msk = np.bitwise_and(msk1, msk2)
    test1 = np.zeros(times_length)
    test1[msk1] = goal
    test1 = test1[msk]
    test1 /= np.std(test1)
    test2 = np.zeros(times_length)
    test2[msk2] = benchmarked
    test2 = test2[msk]
    test2 /= np.std(test2)
    result = np.cov(test1, test2)
    return result[1, 0]


if __name__ == '__main__':

    if parameters['init']:
        if dataset == 'fred':
            import datasets.fred.unified_format as data_source

            data_source.initialize(correlation)
            with open('initialized.pickle', 'wb') as f:
                pk.dump(data_source.as_TSS.as_dict(), f)
        if dataset == 'yahoo':
            import datasets.yahoo.unified_format as data_source

            data_source.init_pickles('10y', '1d')
            data_source.as_TSS.ZAR.clear()
            with open('initialized.pickle', 'wb') as f:
                pk.dump(data_source.as_TSS.as_dict(), f)
    if parameters['log100_removetrend']:
        DAYS = 365 * 10
        if True:
            period = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=DAYS), pd.Timestamp.now(tz='UTC'))
            with open('initialized.pickle', 'rb') as f:
                data_source.as_TSS.load_dict(pk.load(f))

            fake_usd = TimeSeriesSingle()
            for _ in period: fake_usd.append((_, 1.))
            _ = data_source.as_TSS.as_dict()
            _['USD'] = fake_usd

            merged = TimeSeriesMerged(period, _)
            mergedlogged = merged.log100.remove_trend()
            with open('mergedlogged.pickle', 'wb') as f:
                pk.dump(mergedlogged, f)

    with open('mergedlogged.pickle', 'rb') as f:
        mergedlogged: TimeSeriesMerged = pk.load(f)

    high_passed: TimeSeriesMerged = high_passed(mergedlogged)
    for currency, curve in zip(high_passed.currencies, high_passed.as_np):
        comeback = np.concatenate((curve, np.flip(curve, axis=0)))
        freqs = np.fft.fftfreq(len(comeback), d=1)
        filter = np.abs(freqs)
        comeback = np.fft.ifft(np.fft.fft(comeback) * filter)
        curve = comeback[:len(high_passed.merged_times)]
        high_passed[currency] = curve.real

    MAX_EV = len(mergedlogged.currencies) - 1
    PROFIT_RANGES = np.linspace(0.2, 8.0, 27)
    ev, P, Pinv, covar = svd(high_passed)
    sev = np.sqrt(ev)

    volitility = pd.DataFrame(P[:, :MAX_EV], index=mergedlogged.currencies,
                              columns=[f'SE{f:.2e}' for f in sev[:MAX_EV]])
    print(' | '.join(f'{x:.2e}' for x in sev))
    print(volitility)

    curves = []

    times_length = len(mergedlogged.merged_times)
    goals = []
    goals_hf = []
    for i, c in enumerate(Pinv @ high_passed.as_np):
        if i == Pinv.shape[0] - 1: break  # static eigen mode of power 0 [1,1,1,1,1]
        msk, goal = derivate(c)
        goals_hf.append((msk, goal))
    for i, c in enumerate(Pinv @ mergedlogged.as_np):
        if i == Pinv.shape[0] - 1: break  # static eigen mode of power 0 [1,1,1,1,1]
        msk, goal = derivate(c)
        goals.append((msk, goal))

    predictability = np.zeros((9, 10))
    hf = False
    benchmarkeds = []
    _CURVES, _GOAL = (Pinv @ high_passed.as_np, goals_hf) if hf else (Pinv @ mergedlogged.as_np, goals)
    for i, c in enumerate(_CURVES):
        if i == Pinv.shape[0] - 1: break  # static eigen mode of power 0 [1,1,1,1,1]
        msk, benchmarked = np.ones(times_length, dtype=bool), c
        benchmarkeds.append((msk, benchmarked))

    for (i, (msk1, goal)), (j, (msk2, benchmarked)) in \
            itertools.product(enumerate(_GOAL), enumerate(benchmarkeds)):
        predictability[i, j] = profitability(msk1, goal, msk2, benchmarked, times_length)
    for L in predictability:
        print(' '.join(f'{x:+.4f}' for x in L))

    for i, c in enumerate(Pinv @ mergedlogged.as_np):
        if i == Pinv.shape[0] - 1: break
        # benchmarkeds = []
        results = []
        for profit in PROFIT_RANGES:
            # for i, c in enumerate(Pinv @ high_passed.as_np):
            if i == Pinv.shape[0] - 1: break  # static eigen mode of power 0 [1,1,1,1,1]
            msk2, benchmarked = gridify(c, profit=profit)
            # benchmarkeds.append((msk2, benchmarked))
            msk1, goal = goals[i]
            # for (i, (msk1, goal)), (j, (msk2, benchmarked)) in \
            #         zip(enumerate(goals), enumerate(benchmarkeds)):
            res = profitability(msk1, goal, msk2, benchmarked, times_length)
            results.append(res)
            # if (i == j):
            #     print('\t' * (i == j) + f'predictibility of {i} by {j}[profit range = {profit}] = {res}')
        pl.figure()
        pl.ylim(np.min(results), 0.)
        pl.plot(PROFIT_RANGES, results)
    pl.show()

    efficiency = []

    pl.figure()
    pl.title('lowcut')
    for i, c in enumerate(Pinv @ (mergedlogged.as_np - high_passed.as_np)):
        # if i == 0: continue
        if i == MAX_EV: break
        pl.plot(mergedlogged.merged_times, c)
    pl.legend([f'E{x:.2e}' for x in sev][:MAX_EV])

    pl.figure()
    pl.title('high_passed')
    for i, c in enumerate(Pinv @ high_passed.as_np):
        # if i == 0: continue
        if i == MAX_EV: break
        pl.plot(high_passed.merged_times, c)
    pl.legend([f'E{x:.2e}' for x in sev][:MAX_EV])
    pl.show()
    # fred.as_TSS.JPY.plot()
    # pl.show()
    print()
