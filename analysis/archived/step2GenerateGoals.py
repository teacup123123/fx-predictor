from typing import Tuple, Dict

import numpy as np
import pandas as pd

from analysis.parameters import AnalysisParameters, TYPICAL
import pickle as pk

from curves import tools
from curves.t_rate_list_format import TimeSeriesMerged, TimeSeriesSingle


def generateGoals(mergedlogged: TimeSeriesMerged, playables):
    goals_intrinsic = {}
    for a, b in playables:
        if a > b: a, b = b, a
        c = mergedlogged[b] - mergedlogged[a]
        goals_intrinsic[(a, b)] = tools.derivate(c)
    return goals_intrinsic


def main():
    from step0LoadPlayables import loadPlayables
    from step0LoadData import loadData
    from step1LogPercentRemoveTrend import logPercentRemoveTrend

    playables = loadPlayables()
    mergedlogged = logPercentRemoveTrend(loadData())
    return playables, mergedlogged, generateGoals(mergedlogged, playables)


if __name__ == '__main__':
    main()
