import collections
import dataclasses
import math
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from bots.Position import Position, PositionStatus, NatureBuyOrSell


@dataclasses.dataclass
class BotBasket:
    currencies: List[str]

    _time: pd.Timestamp

    @property
    def time(self):
        return self._time

    values_USD_based: np.ndarray

    positions_pendingOrOpen: List[Position]
    positions_Closed: List[Position]
    # route2PosPendingOrOpen_heap: Dict[Tuple[str, str], List]

    swaps: Dict[Tuple[str, str], float]
    lotsize_dict: Dict[Tuple[str, str], Tuple[float, str]]
    deposit_home: float = 0.
    home_currency: str = 'JPY'

    def __init__(self, currencies: List[str], time0: pd.Timestamp, values_USD_based: np.ndarray,
                 lotsize_dict, swaps=None):
        self.currencies = currencies

        self._time = time0

        self.values_USD_based = values_USD_based

        self.positions_pendingOrOpen = []
        self.positions_Closed = []
        # self.route2PosPendingOrOpen_heap = collections.defaultdict(list)

        self.swaps = swaps if swaps is not None else collections.defaultdict(float)
        self.lotsize_dict = lotsize_dict

    def update_time_and_rates(self, new_time: pd.Timestamp, new_values_USD_based: np.ndarray):
        if new_time <= self._time:
            raise ValueError('time only moves forward!')
        delta_time: pd.Timedelta = new_time - self._time

        old_values_USD_based: np.ndarray = np.array(new_values_USD_based)

        new_positions_pendingOrOpen = []
        for position in self.positions_pendingOrOpen:
            idx_base = self.currencies.index(position.base)
            idx_quote = self.currencies.index(position.quote)
            old_rate = old_values_USD_based[idx_base] / old_values_USD_based[idx_quote]
            new_rate = new_values_USD_based[idx_base] / new_values_USD_based[idx_quote]

            if position.status == PositionStatus.pending:
                if (new_rate - position.price_open) * (old_rate - position.price_open) <= 0:
                    position.open(new_time - delta_time * 0.6, position.price_open)
                    if (new_rate - position.price_close) * (1 if position.nature == NatureBuyOrSell.buy else -1) >= 0:
                        position.close(new_time - delta_time * 0.4, position.price_close)
                        self.positions_Closed.append(position)
                    else:
                        new_positions_pendingOrOpen.append(position)
            else:
                assert position.status == PositionStatus.open
                # calculate swap
                swap = self.swaps[(position.sell, position.buy)] * delta_time / pd.Timedelta(years=1)
                position.price_open = position.price_open * \
                                      (1 + swap * (-1 if position.nature == NatureBuyOrSell.buy else +1))

                if (new_rate - position.price_close) * (old_rate - position.price_close) <= 0:
                    position.close(new_time - delta_time * 0.5, position.price_close)
                    self.positions_Closed.append(position)
                else:
                    new_positions_pendingOrOpen.append(position)

        self.positions_pendingOrOpen = new_positions_pendingOrOpen

    def if_open(self, pair: Tuple[str, str], if_open: float, if_close: float):
        sell, buy = pair
        lotsize, denomination = self.lotsize_dict[pair]
        created = Position(buy=buy,
                           sell=sell,
                           lot_currency=denomination,
                           lot_size=lotsize,
                           price_open=if_open,
                           price_close=if_close)
        self.positions_pendingOrOpen.append(created)

    def order_now(self, pair: Tuple[str, str], if_close):
        sell, buy = pair
        lotsize, denomination = self.lotsize_dict[pair]

        created = Position(buy=buy,
                           sell=sell,
                           lot_currency=denomination,
                           lot_size=lotsize,
                           price_open=math.nan,
                           price_close=if_close)

        idx_base = self.currencies.index(created.base)
        idx_quote = self.currencies.index(created.quote)
        open_price = self.values_USD_based[idx_base] / self.values_USD_based[idx_quote]
        created.open(self._time, open_price)

        self.positions_pendingOrOpen.append(created)

    def filter(self, pair: Tuple[str, str], status=None, respect_sell_buy=False):
        def filter_function(position: Position):
            sb = (position.sell, position.buy)
            if status is not None:
                if position.status != status: return False
            if respect_sell_buy:
                if sb != pair:
                    return False
            else:
                a, b = pair
                if sb not in ((a, b), (b, a)):
                    return False
            return True

        return [p for p in self.positions_pendingOrOpen if filter_function(p)]
