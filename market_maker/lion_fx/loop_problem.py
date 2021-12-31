import copy
import math
import os.path
import pickle
import re
from collections import defaultdict
from typing import List, Dict

import pandas as pd

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
    # units in pips per 3 days

    for column in df.columns:
        if column != 'date':
            base, quote, nature = re.match('(...)/(...)_(\w+)', column).groups()
            df[column] *= 1 / (lotsize[f'{base}/{quote}'] * currency_to_jpy[base]) * currency_to_jpy[quote]

    # one week average:
    last = df.iloc[-5:]
    weekly = last.sum()
    # print(weekly)

    for column, value in weekly.items():
        if re.match('(...)/(...)_(\w+)', column):
            base, quote, nature = re.match('(...)/(...)_(\w+)', column).groups()
            if nature == 'buy':
                links[quote][base] = value * 365 / 7  # annual
                print((base, quote, nature), quote, base, value * 365 / 7)
            else:
                links[base][quote] = value * 365 / 7  # annual
                print((base, quote, nature), base, quote, value * 365 / 7)

    if topickle:
        with open('graph.pickle', 'wb') as f:
            pickle.dump((nodes, links), f)
    return nodes, links


def no_positive_loops():
    with open('graph.pickle', 'rb') as f:
        nodes, links = pickle.load(f)

    nodes: List
    links: Dict[str, Dict]
    interdiction = ['TRY', 'ZAR']
    for _ in interdiction:
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

    profitable = (-math.inf, -1, -1)
    for a in nodes:
        for b in nodes:
            if b != a:
                forward = final_graph[a][b]
                back = links[b][a]
                loop_distance = 1
                if (loop_distance, a, b) < profitable:
                    profitable = (loop_distance, a, b)
    print(profitable)


if __name__ == '__main__':
    # build_problem()
    no_positive_loops()
