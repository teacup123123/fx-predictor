import dataclasses
import math
import os.path
from typing import Tuple, Dict

import pickle as pk
import numpy as np

import analysis
from analysis.parameters import AnalysisParameters, DEFAULT

from curves import tools
from curves.t_rate_list_format import TimeSeriesMerged

_, OUTPUT = os.path.split(__file__)
OUTPUT = OUTPUT.replace('.py', '.pickle')


@dataclasses.dataclass
class profitability:
    thresh: float
    x: float
    xlow: float
    xhi: float


def optimizeProfits(mergedlogged: TimeSeriesMerged, playables,
                    params: AnalysisParameters = None):
    if params is None:
        params = DEFAULT

    PROFIT_RANGES = params.p2OptimizeProfitRange.PROFIT_RANGES
    percentage = params.p2OptimizeProfitRange.percentage

    basis = []
    msgs = []
    result = {}
    if params.doStep2COptimizeProfitRange:
        for quote, base in playables:
            curve = mergedlogged[base] - mergedlogged[quote]
            quote_idx, base_idx = map(mergedlogged.currencies.index, (quote, base))
            _ = np.zeros(len(mergedlogged.currencies))
            _[quote_idx] = -1
            _[base_idx] = 1
            basis.append(_)

            accus = []
            accusAbs = []
            closeable_profit = []
            aloss = []

            for x in params.p2OptimizeProfitRange.PROFIT_RANGES:
                msk, benched, accu, accuAbs = tools.gridify(curve, x, accumulated_mvt=True, initIsMean=False)
                accus.append(accu)
                accusAbs.append(accuAbs)
                closeable_profit.append((accuAbs - np.abs(accu)) * x / 2)
                aloss.append(accu ** 2 / 2)
                # qtty = C(qtty/percent) * percent
                # qtty * percent = unrealized loss
                # (qttyAbs - qtty) * profit margin = realized gain = |C| * (accuAbs - accu) * pm
                # unclosable loss = qtty[accu] * accu = C * accu^2

            thresh = percentage * np.max(closeable_profit)
            xlow, xhi = -math.inf, math.inf
            plow, phi = 0., 0.
            for xhi, phi in zip(reversed(PROFIT_RANGES), reversed(closeable_profit)):
                if phi > thresh: break
            for xlow, plow in zip(PROFIT_RANGES, closeable_profit):
                if plow > thresh: break

            x = PROFIT_RANGES[np.argmax(closeable_profit)]
            msg = f'{quote}->{base}: ' \
                  f'{x:.1f}={np.max(closeable_profit):.1f} | ' \
                  f'[{xlow:.2f}~{xhi:.2f}]=[{plow:.1f}~{phi:.1f}]'
            msgs.append(msg)
            print(msg)
            quote: str
            base: str
            result[(quote, base)] = profitability(thresh, x, xlow, xhi)
        with open(os.path.join(analysis.pickles_root, OUTPUT), 'wb') as f:
            pk.dump((basis, result, msgs), f)
    else:
        with open(os.path.join(analysis.pickles_root, OUTPUT), 'rb') as f:
            basis, result, msgs = pk.load(f)

    return basis, result, msgs


def main():
    import step0LoadPlayables, step1LogPercentRemoveTrend
    playables = step0LoadPlayables.loadPlayables()
    mergedlogged = step1LogPercentRemoveTrend.main()
    DEFAULT.doStep2COptimizeProfitRange = True
    optimizeProfits(mergedlogged, playables)


if __name__ == '__main__':
    main()
