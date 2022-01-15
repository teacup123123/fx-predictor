import copy
import math
import os.path
import pickle
import re
import collections
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.optimize

_dir, _ = os.path.split(__file__)

data_lionfx_swap = r'data_lionfx_swap.csv'  # manually downloaded from https://hirose-fx.co.jp/swap/lionfx_swap.csv
data_lotsize = r'data_lotsize.txt'
data_currency_now = r'data_currency_now.txt'


def build_problem(topickle=True):
    links = defaultdict(dict)

    with open(data_lotsize, 'r') as _:
        lotsize = {
            k: float(v.replace(',', '')) for k, v in [line.split() for line in _.readlines()]
        }

    with open(data_currency_now, 'r') as _:
        currency_to_jpy = {
            k: float(v.replace(',', '')) for k, v in [line.split() for line in _.readlines()]
        }
        nodes = list(currency_to_jpy.keys())

    df = pd.read_csv(data_lionfx_swap)

    # units in pips (0.001 movement in the pair)
    for column in df.columns:
        if column != 'date':
            base, quote, nature = re.match('(...)/(...)_(\w+)', column).groups()
            # lot_volume_jpy = lotsize[f'{base}/{quote}'] * currency_to_jpy[base]
            base_quote_rate = (currency_to_jpy[base] / currency_to_jpy[quote])

            pip_rate = 1 / lotsize[f'{base}/JPY']
            df[column] *= pip_rate / (currency_to_jpy[base] / currency_to_jpy[quote])

            # one week average:
    last = df.iloc[-5:]
    weekly = last.sum()
    # print(weekly)

    for column, value in weekly.items():
        if re.match('(...)/(...)_(\w+)', column):
            base, quote, nature = re.match('(...)/(...)_(\w+)', column).groups()
            if nature == 'buy':
                links[quote][base] = value * 365 / 7  # annual
                print((base, quote, nature), value * 365 / 7)
            else:
                links[base][quote] = value * 365 / 7  # annual
                print((base, quote, nature), value * 365 / 7)

    # raise Exception
    if topickle:
        with open('graph.pickle', 'wb') as f:
            pickle.dump((nodes, links), f)
    return nodes, links


def no_positive_loops():
    with open('graph.pickle', 'rb') as f:
        nodes, links = pickle.load(f)

    nodes: List
    links: Dict[str, Dict]
    interdiction = ['TRY'] + ['ZAR'] * 1 + ['SEK', 'NOK', 'SGD', 'HKD', 'TRY', 'HUF', 'PLN'] * 1
    for _ in set(interdiction):
        nodes.remove(_)
        del links[_]
        for k, v in links.items():
            if _ in v:
                del v[_]

    final_graph = {start: {end: (-math.inf,) for end in nodes} for start in nodes}
    for start in nodes:
        final_graph[start][start] = (0., 0, [start])

    goods = []
    for i in range(len(nodes)):
        next_final = copy.deepcopy(final_graph)
        updates = 0
        for p0 in nodes:
            for p1, temp in final_graph[p0].items():
                if len(temp) == 1: continue  # disconnected
                (v01, _, traj01) = temp
                for p2, v12 in links[p1].items():
                    if next_final[p0][p2] < (v01 + v12, -1, -1):
                        updates += 1
                        next_final[p0][p2] = (v01 + v12, 0, traj01 + [p2])
                        traj02 = next_final[p0][p2][-1]
                        if v01 + v12 > 0 and p0 == p2:
                            goods.append(((v01 + v12) / len(traj01), v01 + v12, traj02))
                        if p0 in links[p2]:
                            v20 = links[p2][p0]
                            goods.append(
                                ((v01 + v12 + v20) / (len(traj01) + 1), (v01 + v12 + v20), traj02))

        final_graph = next_final
        if updates == 0: break

    goods = {int(v * 1_000_000): rest for v, *rest in goods}
    goods = [(v / 1_000_000,) + tuple(rest) for v, rest in goods.items()]
    goods.sort()

    for g in goods:
        print(g)

    # goods = list(filter(lambda x: x[0] > -0.0015, goods))
    with open('cycles.pickle', 'wb') as f:
        pickle.dump(goods, f)


def guess_internalv2(threshold=-0.03):
    with open('cycles.pickle', 'rb') as f:
        frees = pickle.load(f)
    with open('graph.pickle', 'rb') as f:
        nodes, links = pickle.load(f)

    def load_identity():
        identity = {}
        id = 0
        for a in links:
            for b in links[a]:
                identity[(a, b)] = id
                id += 1
        return identity

    identity = load_identity()
    link_numbers = len(identity)
    sum_mat, sum_val, vectors = [], [], []
    frees.append((-0.004823, -0.01929374573029336, ['JPY', 'ZAR', 'USD', 'AUD']))
    frees.append((-0.006112, -0.0244490380745789, ['JPY', 'ZAR', 'USD', 'NZD']))
    frees.append((-0.003491, -0.013967369436944961, ['JPY', 'ZAR', 'USD', 'GBP']))
    for interest, totinterest, ll in frees:
        vec = [0. for _ in range(link_numbers)]
        for i, a in enumerate(ll):
            b = ll[(i + 1) % len(ll)]
            id = identity[(a, b)]
            vec[id] = 1.
        vectors.append(vec)
        sum_mat.append(vec)
        sum_val.append(totinterest)
    sum_mat = np.array(sum_mat)
    sum_val = np.array(sum_val)
    vectors = np.array(vectors).T
    ys = sum_mat @ vectors
    solution = vectors[:, :] @ scipy.linalg.solve(ys[:, :], sum_val[:])
    manual_correction = collections.defaultdict(float, {
        'USD': 0.007 - 0.0003,
        'EUR': 0.002 - 0.002 + 0.004,
        'NZD': 0.0009 + 0.0044 - 0.00041,
        'AUD': 0.001 + 0.0036,
        'CHF': 0.0001 + 0.004,
        'GBP': 0.0001 + 0.0045,
        'CAD': 0.004 - 0.002 + 0.0025,
        'MXN': 0.0035,
        'JPY': -0.005 - 0.004 + 0.00065 + 0.0081 + 0.0001
    })
    forbidden = ['SEK', 'NOK', 'SGD', 'HKD', 'TRY', 'HUF', 'PLN', 'ZAR']

    with open('figurative_swap2.txt', 'w') as f:
        f.write('\n'.join(
            f'{a}->{b}:'
            f'{(solution[i] - manual_correction[a] + manual_correction[b]) * 100:+02.2f}%'
            for (a, b), i in identity.items()
            if not any(x in forbidden for x in [a, b])
        ))
    with open('playable.txt', 'w') as f:
        result = []
        for (a, b), i in identity.items():
            val = (solution[i] - (manual_correction[a] if a in manual_correction else 0.) + (
                manual_correction[b] if b in manual_correction else 0.))
            if not any(x in forbidden for x in [a, b]) and val > threshold:
                result.append(f'{a}->{b}:{val * 100:+02.2f}%')
        f.write('\n'.join(result))
    print()


def main():
    # build_problem()
    no_positive_loops()
    # guess_internal()
    guess_internalv2()


if __name__ == '__main__':
    main()
