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
import re
import collections
import numpy as np
import pandas as pd
import pickle as pk
import pylab as pl

import analysis.parameters as ap
from curves import tools
from curves.t_rate_list_format import TimeSeriesMerged, TimeSeriesSingle

ap.DEFAULT = ap.AnalysisParameters(
    doStep0LoadData=False,
    p0LoadData=ap.Step0Params(
        currencies_space_delimited_str='EUR CHF JPY NZD GBP USD CAD mxn aud zar',
        dataset='yahoo'
    ),
    doStep1LogPercentRemoveTrend=True,
    doStep2Correlation=True,
    p2Correlation=ap.Step2CParams(doOrigininal=True, hipassOriginal=9 * 30, doHighPassed=True),
    p1LogPercentRemoveTrend=ap.Step1Params(),
    doStep2COptimizeProfitRange=False,
    p2OptimizeProfitRange=ap.Step2OParams(PROFIT_RANGES=np.arange(0.2, 8.0, 0.25))
)


def main():
    import step0LoadPlayables
    import step0LoadData
    import step1LogPercentRemoveTrend
    import step2OptimizeProfitRange
    import step2Correlation

    playables = step0LoadPlayables.loadPlayables()
    mergedlogged = step1LogPercentRemoveTrend.logPercentRemoveTrend(step0LoadData.loadData())
    basis, profit_table, msgs = step2OptimizeProfitRange.optimizeProfits(mergedlogged, playables)
    _ = step2Correlation.correlation(mergedlogged)
    highflatCorr, lowflatCorr, highlinCorr = _[0], _[1], _[2]

    basis = np.array(basis).T
    risk_focus = lowflatCorr
    P = risk_focus.P
    sev = risk_focus.sev
    profits = np.array([profit_table[pair].thresh for pair in playables])

    noisebasis = []
    for _ in basis.T:
        _ = _.reshape((1, _.size))
        noisebasis.append(np.diag(sev) @ P.T @ _.T @ _ @ P @ np.diag(sev))
    noisebasis = np.array(noisebasis)

    def risk_per_profit(XDirections, verbose=False):
        XDirections /= np.sum(np.abs(XDirections))
        profit = profits @ np.abs(XDirections)
        # X9 = np.random.randn(9) + [0]
        # sum_L(|L @ P @ X9|^2)
        R = np.sum(noisebasis * XDirections.reshape((XDirections.size, 1, 1)), axis=0, keepdims=False)
        u, s, vh = np.linalg.svd(R)
        result = s / profit
        if verbose:
            print(f'{np.sum(result):.2e} | {np.max(result):.2e} | {" ".join(f"{float(f):.1e}" for f in result)}')
        return np.sum(result)

    from market_maker.lion_fx.elementary_cycles import elementary_cycles
    basis_elementary_cycles = []
    for cycle in elementary_cycles:
        _ = np.zeros(len(playables))
        for dir in zip(cycle, cycle[1:] + [cycle[0]]):
            idx = -1
            quote, base = -1, -1
            for idx, (quote, base) in enumerate(playables):
                if (quote, base) == dir: break
            assert (quote, base) == dir
            _[idx] = 1.
        basis_elementary_cycles.append(_)
    basis_elementary_cycles = np.array(basis_elementary_cycles).T

    def risk_per_profit_cycles(Xcycle, verbose=False):
        Y = basis_elementary_cycles @ Xcycle
        return risk_per_profit(Y, verbose)

    directions = collections.defaultdict(bool)
    from scipy.optimize import fmin
    performed = []
    for i in range(10000):
        if i == 0:
            Xseed = np.ones(basis_elementary_cycles.shape[1])
        elif i == 1:
            Xseed = np.zeros(basis_elementary_cycles.shape[1])
            Xseed[2] = 0.22
            Xseed[4] = 0.05
        else:
            Xseed = np.random.rand(basis_elementary_cycles.shape[1]) * 2 - 1
        Xopti = fmin(risk_per_profit_cycles, Xseed, disp=0)
        # Xopti = np.abs(Xopti)
        Xopti /= np.sum(Xopti)
        val = risk_per_profit_cycles(Xopti, verbose=True)
        performed.append((val, Xopti))
    performed.sort()
    print(performed[0])
    final = collections.defaultdict(float)
    _, Xopti = performed[0]
    Yopti = basis_elementary_cycles @ Xopti
    for x, (quote, base) in zip(Yopti, playables):
        directions[(quote, base)] = True
        if final[(quote, base)] - final[(base, quote)] + x > 0:
            final[(quote, base)] = final[(quote, base)] - final[(base, quote)] + x
            final[(base, quote)] = 0
        else:
            final[(base, quote)] = final[(base, quote)] - final[(quote, base)] - x
            final[(quote, base)] = 0
    _ = sum(abs(x) for x in final.values())
    for (a, b) in final:
        final[(a, b)] *= 100. / _
    for (a, b), x in final.items():
        if x > 0:
            if x > 1.0:
                _ = ("<" * (((b, a) in directions))) + "-" + (">" * ((a, b) in directions))
                # print(f'{x :.1f}% x {a}{_}{b}')
                print(f'{x :.1f}% x {a}->{b} @ swap = {playables[(a, b)]}')
        else:
            assert x == 0.



if __name__ == '__main__':
    main()
