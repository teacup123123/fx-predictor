import dataclasses
import enum
import math
from typing import ClassVar

import pandas as pd


class NatureBuyOrSell(enum.Enum):
    buy = enum.auto()
    sell = enum.auto()


class PositionStatus(enum.Enum):
    pending = enum.auto()
    open = enum.auto()
    closed = enum.auto()


@dataclasses.dataclass
class Position:
    """mental health-aware accounting convention for app-users"""
    buy: str
    sell: str
    base: str
    lot_size: float

    @property
    def price_open(self):
        return self._price_open

    @price_open.setter
    def price_open(self, value):
        assert self.status == PositionStatus.pending
        self._price_open = value

    price_close: float

    _price_open: float = math.nan
    _time_opened: pd.Timestamp = None
    _time_closed: pd.Timestamp = None

    id: int = -1
    instances: ClassVar[int] = 0

    def __init__(self,
                 buy: str,
                 sell: str,
                 lot_currency: str,
                 lot_size: float,
                 price_open: float,
                 price_close: float):
        self.buy = buy
        self.sell = sell
        self.base = lot_currency
        self.lot_size = lot_size
        self.price_close = price_close
        self._price_open = price_open
        self.id = Position.instances
        Position.instances += 1

    @property
    def time_opened(self):
        return self._time_opened

    @property
    def time_closed(self):
        return self._time_closed

    def open(self, at_time: pd.Timestamp, price: float) -> None:
        assert self.status == PositionStatus.pending
        self._time_opened = at_time
        self._price_open = price

    def close(self, at_time: pd.Timestamp, price: float) -> float:
        """still need to multiply by currency"""
        if at_time >= self.time_opened:
            self._time_closed = at_time
            self.price_close = price
            return self.unrealized(price)
        else:
            raise ValueError('close can only happen after opening')

    def unrealized(self, price: float):
        return (price - self.price_open) \
               * (+1 if self.nature == NatureBuyOrSell.buy else -1) \
               * self.lot_size  # just need to multiply by currency value of self.quote

    @property
    def quote(self):
        if self.buy == self.base:
            return self.sell
        else:
            return self.buy

    @property
    def nature(self):
        if self.buy == self.base:
            return NatureBuyOrSell.buy
        else:
            return NatureBuyOrSell.sell

    @property
    def status(self):
        if self._time_opened is None:
            return PositionStatus.pending
        elif self._time_closed is None:
            return PositionStatus.open
        else:
            return PositionStatus.closed

    def __repr__(self):
        return f'{self.base}/{self.quote} {self.nature.name} ' \
               f'({self.status}:{self.price_open:.5f}->{self.price_close:.5f})[#{self.id}]'
