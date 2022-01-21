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
import math
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
    lc = len(mergedlogged.currencies)
    basis, profit_table, msgs = step2OptimizeProfitRange.optimizeProfits(mergedlogged, playables)
    _ = step2Correlation.correlation(mergedlogged)
    highflatCorr, lowflatCorr, highlinCorr = _[0], _[1], _[2]

    basis = np.array(basis).T
    risk_focus = lowflatCorr
    P = risk_focus.P
    sev = risk_focus.sev
    profits = np.array([profit_table[pair].thresh for pair in playables])

    pair_risk = {}
    for pair in playables:
        a, b = pair
        a, b = map(lambda x: mergedlogged.currencies.index(x), (a, b))
        _ = np.zeros((lc, 1))
        _[b] = 1.
        _[a] = -1.
        pair_risk[pair] = np.sqrt(_.T @ (highflatCorr.covar + lowflatCorr.covar) @ _)

    indicesEdges = [i for i, (a, b) in enumerate(playables) if a < b]
    playableEdges = [(a, b) for a, b in playables if a < b]
    conversionEdges = [playableEdges.index((min(a, b), max(a, b))) for a, b in playables]
    profitEdges = np.array([profit_table[pair].thresh for pair in playableEdges])
    basisEdges = basis[:, indicesEdges]
    projection = []
    for i, _ in zip(conversionEdges, playables):
        if playableEdges[i] == _:
            _ = np.zeros(len(indicesEdges))
            _[i] = 1.
        else:
            _ = np.zeros(len(indicesEdges))
            _[i] = -1.
        projection.append(_)
    projection = np.array(projection).T

    noisebasis = []
    for _ in basisEdges.T:
        _ = _.reshape((1, _.size))
        noisebasis.append(np.diag(sev) @ P.T @ _.T @ _ @ P @ np.diag(sev))
    noisebasis = np.array(noisebasis)

    noiseEdgesbasis = []
    for _ in basisEdges.T:
        _ = _.reshape((1, _.size))
        noiseEdgesbasis.append(np.diag(sev) @ P.T @ _.T @ _ @ P @ np.diag(sev))
    noiseEdgesbasis = np.array(noiseEdgesbasis)

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
    basisProjected_elementary_cycles = projection @ basis_elementary_cycles

    def risk_per_profit(XDirections, profits, noisebasis, verbose=False):
        XDirections /= np.sum(np.abs(XDirections))
        profit2 = (profits @ np.abs(XDirections)) ** 2
        XDirections2 = XDirections * XDirections
        R2 = np.sum(noisebasis * XDirections2.reshape((XDirections.size, 1, 1)), axis=0, keepdims=False)
        u, s2, vh = np.linalg.svd(R2)
        result2 = s2 / profit2
        if verbose:
            print(
                f'{np.sum(result2):.2e} [sqrt(risk) = {np.sqrt(np.sum(s2)):.2f} %]| '
                f'{np.max(result2):.2e} | '
                f'{" ".join(f"{float(f):.1e}" for f in result2)}')
        return np.sum(result2)

    def optimize(risk_per_profit_cycles, sz, iterations=30, cycle_projection=None, verbose=False):
        from scipy.optimize import fmin
        performed = []
        xoptis = []
        for i in range(iterations):
            if i == 0:
                Xseed = np.ones(sz)
            else:
                Xseed = np.random.rand(sz) * 2 - 1
            Xopti = fmin(risk_per_profit_cycles, Xseed, disp=0)
            val = risk_per_profit_cycles(Xopti, verbose=verbose)
            performed.append((val, i))
            xoptis.append(Xopti / np.sum(Xopti))
        performed.sort()

        Xopti = xoptis[performed[0][1]]

        def analyse_swap(Xopti, verbose=False):
            Yopti = basis_elementary_cycles @ \
                    (Xopti if cycle_projection is None else cycle_projection @ Xopti)
            Yopti *= 2 / np.sum(np.abs(Yopti))
            final = collections.defaultdict(float)
            for x, (quote, base) in zip(Yopti, playables):
                if final[(quote, base)] - final[(base, quote)] + x > 0:
                    final[(quote, base)] = final[(quote, base)] - final[(base, quote)] + x
                    final[(base, quote)] = 0
                else:
                    final[(base, quote)] = final[(base, quote)] - final[(quote, base)] - x
                    final[(quote, base)] = 0
            _ = sum(abs(x) for x in final.values())
            for (a, b) in final:
                final[(a, b)] *= 100. / _
            tot_swap, mx_std = 0., 0.
            for (a, b), x in final.items():
                if x > 0:
                    if x > 1.0:
                        # _ = ("<" * (((b, a) in directions))) + "-" + (">" * ((a, b) in directions))
                        # print(f'{x :.1f}% x {a}{_}{b}')
                        if verbose:
                            print(f'{x :.2f}% x {a}->{b} '
                                  f'@ swap = {playables[(a, b)]} '
                                  f'@ std = {np.sqrt(pair_risk[(a, b)])} %')
                        mx_std = max(mx_std, np.sqrt(pair_risk[(a, b)]))
                        tot_swap += playables[(a, b)] * x * 0.01
                else:
                    assert x == 0.
            if verbose: print(f'tot_swap = {tot_swap} % per year; max std = {mx_std} %')
            return tot_swap

        best = (-math.inf, None)
        nnz = np.flatnonzero(Xopti)
        for signs in itertools.product(*(((-1, 1),) * len(nnz))):
            _Xopti = np.array(Xopti)
            _Xopti[nnz] *= np.array(signs)
            _ = analyse_swap(_Xopti)
            if _ > best[0]: best = (_, _Xopti)
        Xopti = best[1]
        analyse_swap(Xopti, True)
        risk_per_profit_cycles(Xopti, verbose=True)

        return performed

    best = (math.inf, None)
    for allowed_idx in [(4, 13, 20, 21)]:  # itertools.combinations(list(range(22)), 4):
        print(f'==========VV{allowed_idx}VV============')
        allowed = np.zeros(len(elementary_cycles), dtype=bool)
        for _ in allowed_idx: allowed[_] = True
        maskProjection = np.eye(np.size(allowed))
        maskProjection = maskProjection[:, allowed]

        def risk_per_profit_cycles_masked(Xcycle, verbose=False):
            Y = basisProjected_elementary_cycles @ maskProjection @ Xcycle
            return risk_per_profit(Y, profitEdges, noiseEdgesbasis, verbose)

        performedMasked = optimize(risk_per_profit_cycles_masked, sum(allowed), cycle_projection=maskProjection)
        print(performedMasked[0])
        best = best if best[0] < performedMasked[0][0] else performedMasked[0] + (allowed_idx,)
    print(best)

    def risk_per_profit_cycles(Xcycle, verbose=False):
        Y = basisProjected_elementary_cycles @ Xcycle
        return risk_per_profit(Y, profitEdges, noiseEdgesbasis, verbose)

    # performed = optimize(risk_per_profit_cycles, basisProjected_elementary_cycles.shape[1])
    # print(performed[0])


if __name__ == '__main__':
    main()
