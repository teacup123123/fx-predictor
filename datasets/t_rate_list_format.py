import math
import copy
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import dataclasses
import pylab as pl


@dataclasses.dataclass
class TimeSeriesSingle(List[Tuple[pd.Timestamp, float]]):
    @property
    def np(self):
        """returns a copy!"""
        return np.array(self)

    @property
    def inverse(self):
        result = TimeSeriesSingle()
        for t, f in self:
            result.append((t, 1 / f))
        return result

    @property
    def log100(self):
        result = TimeSeriesSingle()
        base = math.log(1.01)
        for t, f in self:
            result.append((t, math.log(f) / base))
        return result

    def plot(self):
        ts, rates = zip(*self.np)
        pl.figure()
        pl.plot(ts, rates)


@dataclasses.dataclass
class TimeSeriesMerged:
    merged_times: List[pd.Timestamp]
    currencies: List[str]
    as_np: np.ndarray
    __lookup_idx = {}

    @property
    def _lookup_idx(self):
        if len(self.__lookup_idx) == 0:
            for i, c in enumerate(self.currencies):
                self.__lookup_idx[c] = i
        return self.__lookup_idx

    def __getitem__(self, currency: str):
        return self.as_np[self._lookup_idx[currency], :]

    def __setitem__(self, currency: str, value):
        self.as_np[self._lookup_idx[currency], :] = value.reshape((1, value.size))

    def __init__(self, period: Tuple[pd.Timestamp, pd.Timestamp], name2TSS: Dict[str, TimeSeriesSingle]):
        self.merged_times = merged_times = []
        self.currencies = []
        start, end = period
        for currencyname, currency_value_tss in name2TSS.items():
            self._lookup_idx[currencyname] = len(self.currencies)
            self.currencies.append(currencyname)
            for t, _ in currency_value_tss:
                if start <= t <= end: merged_times.append(t)
        merged_times.sort()
        tlast = None
        pruned = []
        for t in merged_times:
            if tlast == t:
                continue
            pruned.append(t)
            tlast = t
        self.merged_times = pruned
        good = np.ones(len(pruned), dtype=bool)
        self.as_np = np.zeros((len(self.currencies), good.size))
        for currencyname, currency_value_tss in name2TSS.items():
            i = 0
            lastval = np.nan
            for t, value in currency_value_tss:
                while pruned[i] < t:
                    self[currencyname][i] = lastval
                    if np.isnan(lastval): good[i] = False
                    i += 1
                self[currencyname][i] = value
                lastval = value
        self.merged_times = [t for t, ok in zip(pruned, good) if ok]
        self.as_np = self.as_np[:, good]

    @property
    def log100(self):
        result = copy.deepcopy(self)
        result.as_np = np.log(result.as_np) / np.log(1.01)
        return result

    def copy(self):
        return copy.deepcopy(self)

    @property
    def remove_trend(self, deg=1):
        result = self.copy()
        ts = [t - result.merged_times[0] for t in result.merged_times]
        ts = np.array([t.value for t in ts]) / (1e9 * 3600 * 24)
        for c, curve in zip(result.currencies, result.as_np):
            fit = np.polyfit(ts, curve, deg)
            offset = 0
            for i, coef in enumerate(reversed(fit)):
                offset += coef * ts ** i
            result[c] = result[c] - offset
        offsets = result.as_np.mean(axis=1)

        return result


if __name__ == '__main__':
    a = TimeSeriesSingle()
    a.append((pd.Timestamp.now(), 1.))
    print(a)
    print(a.np)
    for x in a:
        print(x)
