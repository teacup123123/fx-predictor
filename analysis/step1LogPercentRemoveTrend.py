import os
from typing import Tuple

import pandas as pd

import analysis
from analysis.parameters import AnalysisParameters, DEFAULT
import pickle as pk

from curves.t_rate_list_format import TimeSeriesMerged, TimeSeriesSingle

_, OUTPUT = os.path.split(__file__)
OUTPUT = OUTPUT.replace('.py', '.pickle')


def merged(input, params: AnalysisParameters = None) -> TimeSeriesMerged:
    return calc_all(input, params)[0]


def logPercent(input, params: AnalysisParameters = None) -> TimeSeriesMerged:
    return calc_all(input, params)[1]


def logPercentRemoveTrend(input, params: AnalysisParameters = None) -> TimeSeriesMerged:
    return calc_all(input, params)[2]


def calc_all(input, params: AnalysisParameters = None):
    if params is None:
        params = DEFAULT
    if params.doStep1LogPercentRemoveTrend:
        _, asdict, data_source = input
        period = (
            pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=params.p1LogPercentRemoveTrend.days),
            pd.Timestamp.now(tz='UTC')
        )
        for _, v in data_source.as_TSS.USD:
            assert v == 1.

        fake_usd = TimeSeriesSingle()
        for _ in period: fake_usd.append((_, 1.))
        _ = data_source.as_TSS.as_dict()
        _['USD'] = fake_usd

        merged = TimeSeriesMerged(period, _)
        mergedlogged = merged.log100
        mergedloggedNoTrend: TimeSeriesMerged = mergedlogged.remove_trend()
        with open(os.path.join(analysis.pickles_root, OUTPUT), 'wb') as f:
            pk.dump((merged, mergedlogged, mergedloggedNoTrend), f)
    with open(os.path.join(analysis.pickles_root, OUTPUT), 'rb') as f:
        merged: TimeSeriesMerged
        mergedlogged: TimeSeriesMerged
        mergedloggedNoTrend: TimeSeriesMerged
        merged, mergedlogged, mergedloggedNoTrend = pk.load(f)
    return merged, mergedlogged, mergedloggedNoTrend


def main():
    from step0LoadData import loadData
    DEFAULT.doStep1LogPercentRemoveTrend = True
    return logPercentRemoveTrend(loadData())


if __name__ == '__main__':
    main()
