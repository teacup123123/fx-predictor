import math
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass, fields, field
from typing import List, Callable, Any


class EnumIdx0(Enum):
    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class Currency(EnumIdx0):
    """available currencies"""
    jpy = ()
    cad = ()
    eur = ()
    usd = ()


@dataclass
class Rates:
    """The rates at a specific time, thin wrapper of a list object"""
    as_list: List[float] = field(default_factory=list)
    home_currency: Currency = Currency.jpy

    def __init__(self):
        self.as_list = [0.] * (len(Currency))
        self.home_currency = Currency.jpy
        self[Currency.jpy] = 1.
        self[Currency.eur] = 100.
        self[Currency.cad] = 90.
        self[Currency.usd] = 95.

    def of_currency(self, c: Currency):
        return self.as_list[c.value]

    def __getitem__(self, currency: Currency):
        return self.as_list[currency.value]

    def __setitem__(self, currency: Currency, value):
        self.as_list[currency.value] = value


@dataclass
class CurrencyBasket:
    as_list = [0.] * (len(Currency))

    def unrealized(self, rates: Rates):
        result_home = 0.
        for currency, quantity in self.as_list:
            result_home += quantity * rates.of_currency(currency)
        return result_home

    def __getitem__(self, currency: Currency):
        return self.as_list[currency.value]

    def __setitem__(self, currency: Currency, value):
        self.as_list[currency.value] = value


@dataclass
class Position:
    target_currency: Currency
    base_currency: Currency
    home_currency: Currency

    realized_home: float = 0.
    target_quantity: float = 0.
    base_quantity: float = 0.
    isBuy: bool = True

    def start_position(self):
        return self.target_quantity / self.base_quantity

    def enter(self, lotsize_target: float, rates: Rates):
        sign = +1 if self.isBuy else -1
        quantity = lotsize_target * 1000
        self.target_quantity += sign * quantity
        self.base_quantity -= sign * rates.of_currency(self.target_currency) / rates.of_currency(
            self.base_currency) * quantity

    def unrealized(self, rates: Rates):
        result = 0.
        result += self.target_quantity * rates.of_currency(self.target_currency)
        result += self.base_quantity * rates.of_currency(self.target_currency)
        result /= rates.of_currency(self.home_currency)
        return result

    def exit(self, lotsize_target: float, rates: Rates):
        self.enter(-lotsize_target, rates)
        base_retained = -rates.of_currency(self.target_currency) / rates.of_currency(
            self.base_currency) * self.target_quantity
        redeemed_base = self.base_quantity - base_retained
        self.realized_home += redeemed_base * rates.of_currency(self.base_currency)
        self.base_quantity -= redeemed_base


@dataclass
class PositionAssets:
    positions: List[Position] = field(default_factory=list)

    def currency_basket(self):
        cb = CurrencyBasket()
        for position in self.positions:
            cb[position.target_currency] += position.target_quantity
            cb[position.base_currency] += position.base_quantity
        return cb

    def unrealized(self, rates: Rates):
        return self.currency_basket().unrealized(rates)

    @property
    def realized(self):
        return sum(position.realized_home for position in self.positions)

    @staticmethod
    def decorator_position(position_Creator: Callable[[Any], Position]):
        added_at = -1

        def decorated(self):
            nonlocal added_at
            self: PositionAssets
            if added_at == -1:
                position = position_Creator(self)
                added_at = len(self.positions)
                self.positions.append(position)
            return self.positions[added_at]

        return property(decorated)

        # No setter coz position once defined is final
        # def decorated_setter(self, value):
        #     nonlocal added_at
        #     self: PositionAssets
        #     if added_at == -1:
        #         position = position_Creator(self)
        #         added_at = len(self.positions)
        #         self.positions.append(position)
        #     self.positions[added_at] = value
        # return property(decorated, fset=decorated_setter)
