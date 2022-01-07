import itertools
import math
import pickle as pk
import os
import re
import time
from collections import defaultdict

import numpy as np
import pylab as pl

_dir = r'pickles'

pyramid = 'GBP JPY USD EUR'.split()


def create_folders():
    for file in os.listdir(_dir):
        if file.endswith('.pickle'):
            attempt_re = re.match(f'([A-Z]+)=X_(\w+)_(\w+)', file)
            pair, dur, resolution = attempt_re.groups()
            os.makedirs(f'{dur}_{resolution}', exist_ok=True)


def weave(start, end, duration_str: str = '2y', resolution_str: str = '60m', include_only=None):
    """usd = 1 all the time"""
    good_files = []
    for file in os.listdir(_dir):
        if file.endswith('.pickle'):
            attempt_re = re.match(f'([A-Z]+)=X_(\w+)_(\w+)', file)
            if not attempt_re: continue
            pair, dur, resolution = attempt_re.groups()
            if len(pair) == 3: pair = 'USD' + pair
            base, quote = pair[:3], pair[3:]
            if include_only and any(x not in include_only for x in [base, quote]): continue
            if duration_str != dur or resolution_str != resolution: continue
            good_files.append((file, pair, dur, resolution))

    common_timestamp = []
    for gfi, (file, pair, dur, resolution) in enumerate(good_files):
        with open(os.path.join(_dir, file), 'rb') as f:
            data = pk.load(f)
            data = data['chart']['result'][0]
            # meta = data['meta']
            timestamp0 = data['timestamp'][:-1]
            indicators = data['indicators']
            quotes = indicators['quote'][0]
            high, low = [quotes[_][:-1] for _ in 'high,low'.split(',')]
        if gfi == 0:
            common_timestamp = [
                t for i, t in enumerate(timestamp0)
                if start <= timestamp0[i] < end and all(x is not None for x in (high[i], low[i]))
            ]
        else:
            building = []
            assert min(len(high), len(low), len(timestamp0)) == max(len(high), len(low), len(timestamp0))
            i, j = 0, 0
            while i < len(timestamp0) and j < len(common_timestamp):
                if common_timestamp[j] == timestamp0[i]:
                    if start <= timestamp0[i] < end and all(x is not None for x in (high[i], low[i])):
                        building.append(timestamp0[i])
                    i += 1
                    j += 1
                elif common_timestamp[j] < timestamp0[i]:
                    j += 1
                elif common_timestamp[j] > timestamp0[i]:
                    i += 1
                else:
                    break
            common_timestamp = building
    common_timestamp = set(common_timestamp)
    perms = list(itertools.permutations([0, 1, 2, 3]))
    perm_idx = np.random.randint(0, len(perms), size=len(common_timestamp))

    full_data = {}
    pyramid_by = {}
    ts = []

    for file, pair, dur, resolution in good_files:
        if duration_str != dur or resolution_str != resolution: continue
        base, quote = pair[:3], pair[3:]
        if quote in pyramid and base not in pyramid:
            pyramid_by[base] = quote
        if base in pyramid and quote not in pyramid:
            pyramid_by[quote] = base

        with open(os.path.join(_dir, file), 'rb') as f:
            data = pk.load(f)
            data = data['chart']['result'][0]
            # meta = data['meta']
            timestamp0 = data['timestamp'][:-1]
            indicators = data['indicators']
            quotes = indicators['quote'][0]

        dt = int(np.median(np.diff(timestamp0))) // 4
        high, low = [quotes[_][:-1] for _ in 'high,low'.split(',')]
        assert min(len(high), len(low), len(timestamp0)) == max(len(high), len(low), len(timestamp0))
        high, low, timestamp = zip(*[
            (hi, lo, t) for hi, lo, t in zip(high, low, timestamp0)
            if start <= t < end and all(x is not None for x in (hi, lo, t)) and t in common_timestamp
        ])
        high, low, timestamp = map(np.array, (high, low, timestamp))

        # xored = common_timestamp - set(timestamp)
        ts = []
        sigbase = []
        sigquote = []

        if base in pyramid and quote in pyramid:
            for ti, (t, hi, lo) in enumerate(zip(timestamp, high, low)):
                for i, peak_currency in enumerate(pyramid[pi] for pi in perms[perm_idx[ti]]):
                    ts.append(t + i * dt)
                    _bf, _qf = 1., 1.
                    if peak_currency == base:
                        _bf, _qf = 1., 1. / hi
                    if peak_currency == quote:
                        _bf, _qf = lo, 1.
                    sigbase.append(_bf)
                    sigquote.append(_qf)
        else:
            if base in pyramid:
                base, quote = quote, base
                high, low = [1. / lo for lo in low], [1. / hi for hi in high]
            for ti, (t, hi, lo) in enumerate(zip(timestamp, high, low)):
                # now quote in pyramid
                for i, peak_currency in enumerate(pyramid[pi] for pi in perms[perm_idx[ti]]):
                    ts.append(t + i * dt)
                    sigbase.append(lo if peak_currency == quote else hi)
                    sigquote.append(1.)
        full_data[base] = full_data.get(base, 1.) * np.array(sigbase)
        full_data[quote] = full_data.get(quote, 1.) * np.array(sigquote)
    for k, v in pyramid_by.items():
        full_data[k] *= full_data[v]
    for k in full_data:
        if k != 'USD':
            full_data[k] = full_data[k] / full_data['USD']
    full_data['USD'] = full_data['USD'] * 0 + 1.

    lastTime = defaultdict(float)
    for i in range(len(full_data) // 4):
        best = (math.inf, -1)
        for permutation in perms:
            score = 0.
            for k in full_data:
                last = lastTime[k]
                for pi in permutation:
                    score += np.abs(last - np.log(full_data[k][i * 4 + pi]))
                    last = np.log(full_data[k][i * 4 + pi])
            best = min(best, (score, permutation))

        for k in full_data:
            _, permutation = best
            times = ts[i * 4:i * 4 + 4]
            vals = full_data[k][i * 4:i * 4 + 4]
            vals = [vals[i] for i in permutation]
            for j, t, v in zip(range(i * 4, i * 4 + 4), times, vals):
                full_data[k][j] = v
    return ts, full_data


def append(duration_str: str = '2y', resolution_str: str = '60m', include_only=None):
    start0 = 0
    end0 = int(time.time())
    for fn in os.listdir(_dir):
        match = re.match(f'accu_(\w+)_(\w+)_(\w+)_(\w+).pickle', fn)
        if match:
            duration, resolution, start, end = match.groups()
            if (duration_str, resolution_str) != (duration, resolution):
                continue
            start0 = max(start0, int(end))
    times, full_data = weave(resolution_str=resolution_str, duration_str=duration_str,
                             start=start0, end=end0, include_only=include_only)

    with open(os.path.join(_dir, f'accu_{duration_str}_{resolution_str}_{start0}_{end0}.pickle'), 'wb') as f:
        pk.dump((times, full_data), f)


if __name__ == '__main__':
    # create_folders()

    include_only = '''NZD JPY CAD GBP MXN EUR CHF USD AUD ZAR'''.split()
    for duration_str, resolution_str in [('1mo', '5m'), ('2y', '60m'), ('10y', '1d')]:
        # initially
        start, end = 0, 1640000000
        times, full_data = weave(resolution_str=resolution_str, duration_str=duration_str,
                                 start=start * 0, end=end, include_only=include_only)
        with open(os.path.join(_dir, f'accu_{duration_str}_{resolution_str}_{start}_{end}.pickle'), 'wb') as f:
            pk.dump((times, full_data), f)

        # append
        append(duration_str, resolution_str, include_only)

    # base, quote = 'AUD USD'.split()
    # pl.title(f'{base}/{quote}')
    # pl.plot(times, full_data[base] / full_data[quote])
    # pl.show()
    # print()
