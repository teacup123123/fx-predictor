import dataclasses
import os
from typing import List

import numpy as np
import pandas as pd

import analysis
from analysis.parameters import AnalysisParameters, DEFAULT
import pickle as pk

from curves import tools
from curves.t_rate_list_format import TimeSeriesMerged

_, OUTPUT = os.path.split(__file__)
OUTPUT = OUTPUT.replace('.py', '.pickle')


@dataclasses.dataclass
class corrResult:
    ev: np.ndarray
    sev: np.ndarray
    P: np.ndarray
    Pinv: np.ndarray
    covar: np.ndarray
    volitility: pd.DataFrame


def correlation(mergedlogged: TimeSeriesMerged, params: AnalysisParameters = None) -> List[corrResult]:
    if params is None:
        params = DEFAULT
    p2 = params.p2Correlation
    todos = \
        [*tools.high_pass_cutoff(mergedlogged, p2.hipassOriginal)] * p2.doOrigininal + \
        [tools.high_pass(mergedlogged)] * p2.doHighPassed
    correlationData = []
    if params.doStep2Correlation:
        for high_passed in todos:
            MAX_EV = len(high_passed.currencies) - 1
            ev, P, Pinv, covar = tools.pca(high_passed)
            ev = np.abs(ev)
            sev = np.sqrt(ev)

            volitility = pd.DataFrame(P[:, :MAX_EV], index=mergedlogged.currencies,
                                      columns=[f'SE{f:.2e}' for f in sev[:MAX_EV]])
            print(' | '.join(f'{x:.2e}' for x in sev))
            print(volitility)
            correlationData.append(corrResult(ev, sev, P, Pinv, covar, volitility))

        with open(os.path.join(analysis.pickles_root, OUTPUT), 'wb') as f:
            pk.dump(correlationData, f)
    with open(os.path.join(analysis.pickles_root, OUTPUT), 'rb') as f:
        correlationData = pk.load(f)
    return correlationData


def main():
    from step0LoadData import loadData
    from step1LogPercentRemoveTrend import logPercentRemoveTrend

    DEFAULT.doStep2Correlation = True
    DEFAULT.p2Correlation.doOrigininal = False
    DEFAULT.p2Correlation.doHighPassed = True
    correlation(logPercentRemoveTrend(loadData()))


if __name__ == '__main__':
    main()
