import os.path
import re

import numpy as np
import pandas as pd
import pylab as pl

from bots.BotBasket import BotBasket
from bots.DeadlockBot import DeadlockBot


def create_dlb_bestcycle_masked100(days=365):
    from analysis.step0LoadData import loadData
    from analysis import step1LogPercentRemoveTrend as s1
    from analysis.step0LoadPlayables import loadPlayables
    from analysis.step2OptimizeProfitRange import optimizeProfits
    from market_maker.lion_fx.elementary_cycles import elementary_cycles
    from analysis import parameters
    from market_maker.lion_fx import annualized_swap

    nodes, swaps = annualized_swap.loadSwap()
    playables = loadPlayables()

    parameters.DEFAULT.doStep1LogPercentRemoveTrend = True
    merged = s1.merged(loadData())
    mergedlogged = merged.log100

    coefs_cycles = np.zeros(22)
    coefs_cycles[[1, 8, 13]] = [0.31555707, 0.21215859, 0.47228434]
    coefs_cycles /= np.sum(np.abs(coefs_cycles))
    MAX_OVER_MIN = 5
    while np.min(np.abs(coefs_cycles[coefs_cycles != 0.0])) * MAX_OVER_MIN < np.max(np.abs(coefs_cycles)):
        coefs_cycles[np.abs(coefs_cycles) * MAX_OVER_MIN < np.max(np.abs(coefs_cycles))] = 0.
        coefs_cycles /= np.sum(np.abs(coefs_cycles))
    coefs_cycles /= np.sign(np.sum(coefs_cycles))

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
    coefs_deadlock = basis_elementary_cycles @ np.array(coefs_cycles)

    _ = list(playables.keys())
    for i, v in enumerate(coefs_deadlock):
        if v < 0:
            a, b = _[i]
            coefs_deadlock[_.index((b, a))] += -v
            coefs_deadlock[i] = 0.0
    coefs_deadlock /= np.sum(np.abs(coefs_deadlock))

    simulationEnd = mergedlogged.merged_times[-1]
    simulationStart = mergedlogged.merged_times[-1]
    i = -1
    for i, simulationStart in enumerate(reversed(mergedlogged.merged_times)):
        if simulationStart < simulationEnd - pd.Timedelta(days=days):
            break
    annual_projection = np.mean(merged.as_np[:, -1 - i:-1], axis=1)

    basis, profit_table, msgs = optimizeProfits(mergedlogged, playables)

    dlb = DeadlockBot(merged.currencies,
                      playables,
                      coefs_deadlock,
                      {k: v.xhi * 0.3 + v.xlow * 0.7 for k, v in profit_table.items()},
                      annual_projection)
    return dlb


def main(days=365 * 8):
    from analysis.step0LoadData import loadData
    from analysis import step1LogPercentRemoveTrend as s1
    # for establishing annual projection

    from market_maker import lion_fx
    from market_maker.lion_fx import annualized_swap
    import analysis

    from analysis.step0LoadPlayables import loadPlayables
    from analysis.step2OptimizeProfitRange import optimizeProfits

    nodes, swaps = annualized_swap.loadSwap()
    playables = loadPlayables()

    merged = s1.merged(loadData())
    mergedlogged = merged.log100
    basis, profit_table, msgs = optimizeProfits(mergedlogged, playables)

    lotsize_dict = {}
    with open(os.path.join(lion_fx.root, 'data_lotsize.txt'), 'r') as _:
        for line in _.readlines():
            line = line.strip()
            base, quote, size = re.match('(...)/(...) ([0-9,]+)', line).groups()
            lotsize_dict[(quote, base)] = lotsize_dict[(base, quote)] = (float(size.replace(',', '')), base)

    nodes, swaps = annualized_swap.loadSwap()
    merged = s1.merged(loadData())
    simulationEnd = mergedlogged.merged_times[-1]
    simulationStart = mergedlogged.merged_times[-1]
    i = -1
    for i, simulationStart in enumerate(reversed(mergedlogged.merged_times)):
        if simulationStart < simulationEnd - pd.Timedelta(days=days):
            break
    annual_projection = np.mean(merged.as_np[:, -1 - i:-1], axis=1)

    with open(os.path.join(analysis.root, 'interesting.txt'), 'r') as _f:
        bot_descriptions = _f.read(-1).split('\n\n')

    def from_description(description: str):
        lines = description.splitlines(keepends=False)
        pattern_header = r'==========VV\(([\w ,]+)\)VV============'
        pattern_swap_mxstd = r'tot_swap = ([\-0-9.]+) % per year; max std = \[\[([\-0-9.]+)\]\] %'
        pattern_risk = r'([0-9\-e.]+) \[sqrt\(risk\) = ([0-9\-e.]+) %\]\| ([0-9\-e.]+) \|(( .\..e-..)+)'
        pattern_directions = r'([0-9.]+)% x (\w+)->(\w+) @ swap = ([\-0-9.]+) @ std = \[\[([0-9.]+)\]\] %'

        def single_pattern(pattern):
            res = [re.match(pattern, line) for line in lines]
            res = [r for r in res if r is not None]
            res = [r.groups() for r in res]

            def isnum(f):
                try:
                    float(f)
                    return True
                except ValueError:
                    return False

            res = [tuple(float(f) if isnum(f) else f for f in r) for r in res]
            if len(res) == 1: res = res[0]
            return res

        header, swap_mxstd, risk, directions = \
            map(single_pattern,
                (pattern_header, pattern_swap_mxstd, pattern_risk, pattern_directions)
                )
        coefs = np.zeros(len(playables))
        playables_index = {k: i for i, k in enumerate(playables)}
        for deposit_p, sell, buy, swap, std_p in directions:
            coefs[playables_index[(sell, buy)]] = deposit_p * 0.01
        coefs /= np.sum(coefs)

        dlb = DeadlockBot(merged.currencies,
                          playables,
                          coefs,
                          {k: v.xhi * 0.5 + v.xlow * 0.5 for k, v in profit_table.items()},
                          annual_projection,
                          lever=5 / risk[1])
        HALF_RANGE_IN_STDs = 5
        dlb.grid_min_max_difference_ratio = (2 * HALF_RANGE_IN_STDs * swap_mxstd[-1]) * 0.01
        dlb.name = header[0]
        return dlb

    bots = [from_description(d) for d in bot_descriptions]
    # bots_idx = [0, 1, 11]
    bots_idx = [0, 2]
    bots = [bots[i] for i in bots_idx]
    legends = [bot.name for bot in bots]
    for bot in bots:
        # bot = create_dlb_bestcycle_masked100(days)

        end = -1
        start = end
        while merged.merged_times[end] - merged.merged_times[start] <= pd.Timedelta(days=days):
            start -= 1
        swaps_formated = {}
        for k, v in swaps.items():
            for kk, vv in v.items():
                swaps_formated[(k, kk)] = vv
        bb = BotBasket(merged.currencies, merged.merged_times[start], merged.as_np[:, start], lotsize_dict,
                       swaps_formated)
        bb.deposit_home = 10_000_000
        bot.attach_to_botBasket(bb)

        worst_unrealized_over_deposit = 0
        fulls = []
        for i in range(start + 1, end):
            negatives, positives, full = bb.update_time_and_rates(merged.merged_times[i], merged.as_np[:, i])
            unrealized = negatives + positives
            deposit = full - unrealized
            worst_unrealized_over_deposit = min(worst_unrealized_over_deposit, unrealized / deposit)
            fulls.append(full)
            bot.reeval(merged.as_np[:, i])
        # pl.title(f'worst_unrealized_over_deposit ratio = {worst_unrealized_over_deposit * 100:.2f}%')
        pl.semilogy(merged.merged_times[start + 1: end], fulls)
    pl.legend(legends)
    pl.show()


if __name__ == '__main__':
    main()
