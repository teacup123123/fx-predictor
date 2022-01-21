import dataclasses
import math
from typing import List, Union, Dict, Tuple

import numpy as np

from bots.BotBasket import BotBasket
from bots.Position import PositionStatus, Position


@dataclasses.dataclass
class DeadlockBot:
    """
    the coefficients C, all positive ONLY, determine the relative ``lot_value/%``
    the deadlock value V at equilibrium is propto C, effectively creating a range of half-and half
    * once the deadlock is overruled in the opposite direction, we `could` reverse the nature
    * once the deadlock reaches in the opposite direction, we `could` reverse the nature

    """
    currencies: List[str]
    playables: Union[Dict, List]
    coefficients_directions: np.ndarray
    no_hold_line: np.ndarray
    lever: float
    profit_table: Dict[Tuple[str, str], float]
    attached_to: BotBasket = None

    gridded_deposit_unit: int = 2_0000
    """resolution for compound interest"""

    gridded_deposit_number: int = 0

    grid_min_max_difference_ratio: float = 0.25
    """gridmin = 1 - grid_min_max_difference_ratio/2;  gridmin = 1 + grid_min_max_difference_ratio/2"""

    @property
    def deposit_gridded(self):
        deposit_present = self.attached_to.deposit_home
        if (deposit_present / self.gridded_deposit_unit) > self.gridded_deposit_number + 1:
            # scale-up
            while (deposit_present / self.gridded_deposit_unit) > self.gridded_deposit_number:
                self.gridded_deposit_number += 1  # now smaller
        elif (deposit_present / self.gridded_deposit_unit) < self.gridded_deposit_number - 5:
            print('warning: downsizing')
            # scale-down
            while (deposit_present / self.gridded_deposit_unit) < self.gridded_deposit_number:
                self.gridded_deposit_number -= 1  # now smaller
        return self.gridded_deposit_number * self.gridded_deposit_unit

    def foco_pair(self):
        pass

    def __init__(self,
                 currencies,
                 playables,
                 coefficients_directions,
                 profit_table: Dict[Tuple[str, str], float],
                 no_hold_line, lever=5.):
        self.currencies = currencies
        assert len(playables) == len(coefficients_directions)
        self.coefficients_directions = coefficients_directions
        self.playables = playables
        self.profit_table = profit_table
        assert [k for k in playables] == list(profit_table.keys())
        self.no_hold_line = no_hold_line
        self.lever = lever

    def reeval(self, new_status):
        relative_strength = new_status / self.no_hold_line
        idx_home = self.currencies.index(self.attached_to.home_currency)
        # calculate holdings...
        for c, route, profit_range_percent in \
                zip(self.coefficients_directions, self.playables, self.profit_table.values()):
            if c > 0.0:
                sell, buy = route
                lotsize, lotunit = self.attached_to.lotsize_dict[route]
                idx_sell = self.currencies.index(sell)
                idx_buy = self.currencies.index(buy)
                idx_base = self.currencies.index(lotunit)
                idx_quote = idx_sell + idx_buy - idx_base
                buy_sign = +1 if idx_buy == idx_base else -1

                lot_value_equil = lotsize * self.no_hold_line[idx_base]
                levered_deposit_equil = self.deposit_gridded * self.no_hold_line[idx_home] * self.lever
                lots_equil = int(2 * round(levered_deposit_equil * c / lot_value_equil)) / 2

                readprice_equil = self.no_hold_line[idx_base] / self.no_hold_line[idx_sell + idx_buy - idx_base]
                readprice_profit_range = profit_range_percent * 0.01 * readprice_equil
                readprice_minmax_range = self.grid_min_max_difference_ratio * readprice_equil
                max_lotsize = int(round(2 * lots_equil))
                readprice_trap_range = readprice_minmax_range / max_lotsize
                readprice_equil_profit_range = readprice_equil * profit_range_percent * 0.01
                readprice_zerolots = readprice_equil + lots_equil * readprice_trap_range
                readprice_grid = np.linspace(readprice_equil + readprice_minmax_range / 2 * buy_sign,
                                             readprice_equil - readprice_minmax_range / 2 * buy_sign,
                                             max_lotsize + 1)
                readprice_grid_open = readprice_grid - buy_sign * readprice_equil_profit_range / 2
                readprice_grid_close = readprice_grid + buy_sign * readprice_equil_profit_range / 2

                readprice = new_status[idx_base] / new_status[idx_quote]
                # pair_strength = relative_strength[idx_buy] / relative_strength[idx_sell]

                at_least = np.sum(readprice_grid_open[1:] * buy_sign >= readprice * buy_sign)
                at_most = np.sum(readprice_grid_close[1:] * buy_sign >= readprice * buy_sign)

                positions_opened = self.attached_to.filter(route, status=PositionStatus.open, respect_sell_buy=True)
                positions_opened.sort(key=lambda p: -p.price_open * buy_sign)
                price_idx = 1
                for position, price_idx in zip(
                        positions_opened,
                        range(1, len(readprice_grid_open) + 1)
                ):
                    position.price_close = readprice_grid_close[price_idx]
                    price_idx += 1

                positions_opened = self.attached_to.filter(route, status=PositionStatus.open, respect_sell_buy=True)
                for _ in range(at_least - len(positions_opened)):
                    self.attached_to.order_now(route, readprice_grid_close[price_idx])
                    price_idx += 1

                positions_opened = self.attached_to.filter(route, status=PositionStatus.open, respect_sell_buy=True)
                if len(positions_opened) > at_most:
                    print('this should only happen during shrinking catastrophy!?')

                positions_pending = self.attached_to.filter(route, status=PositionStatus.pending, respect_sell_buy=True)
                positions_pending.sort(key=lambda p: -p.price_open * buy_sign)
                for p in positions_pending:
                    p.price_open = readprice_grid_open[price_idx]
                    p.price_close = readprice_grid_close[price_idx]
                    price_idx += 1
                for _ in range(min(12 - len(positions_pending), readprice_grid_open.size - price_idx)):
                    self.attached_to.if_open(route, readprice_grid_open[price_idx], readprice_grid_close[price_idx])
                    price_idx += 1

        # rush buy if possible, else order IF_OPEN's, establishing IF_CLOSE at the same time

    def attach_to_botBasket(self, botbasket: BotBasket):
        self.attached_to = botbasket
