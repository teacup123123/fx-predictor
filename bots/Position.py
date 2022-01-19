import dataclasses
import enum

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
    lot_currency: str
    lot_size: float

    price_open: float
    price_close: float

    _time_opened: pd.Timestamp = None
    _time_closed: pd.Timestamp = None

    @property
    def time_opened(self):
        return self._time_opened

    @property
    def time_closed(self):
        return self._time_closed

    def open(self, at_time: pd.Timestamp, price: float) -> None:
        assert self.status == PositionStatus.pending
        self._time_opened = at_time
        self.price_open = price

    def close(self, at_time: pd.Timestamp, price: float) -> float:
        if at_time >= self.time_opened:
            self._time_closed = at_time
            self.price_close = price
            return (self.price_close - self.price_open) * self.lot_size
        else:
            raise ValueError('close can only happen after opening')

    @property
    def base(self):
        return self.lot_currency

    @property
    def quote(self):
        if self.buy == self.lot_currency:
            return self.sell
        else:
            return self.buy

    @property
    def nature(self):
        if self.buy == self.lot_currency:
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
        return f'{self.base}/{self.quote} {self.nature.name}'
