import math
from enum import Enum
from dataclasses import dataclass, field
from collections import Sized
from typing import List, Callable, Any, NewType, Union, Iterable, Type


class CurrencyEnumIdx0(Enum):
    """automatic enum structure that starts with index = 0 """

    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


Derived_CurrencyEnumIdx0 = NewType('derived_CurrencyEnumIdx0', CurrencyEnumIdx0)


@dataclass
class Rates:
    """The rates at a specific time, thin wrapper of a list object"""
    home_currency: Derived_CurrencyEnumIdx0
    as_list: List[float] = field(default_factory=list)

    def __init__(self, currency_enum, home_currency: Derived_CurrencyEnumIdx0):
        self.as_list = [math.nan] * (len(currency_enum))
        self.home_currency = home_currency

    def of_currency(self, c: Derived_CurrencyEnumIdx0):
        return self.as_list[c.value]

    def __getitem__(self, Derived_CurrencyEnumIdx0: Derived_CurrencyEnumIdx0):
        return self.as_list[Derived_CurrencyEnumIdx0.value]

    def __setitem__(self, Derived_CurrencyEnumIdx0: Derived_CurrencyEnumIdx0, value):
        self.as_list[Derived_CurrencyEnumIdx0.value] = value


@dataclass
class CurrencyBasket:
    """A basket with different currencies of different interest rates"""
    _CurrencyEnumIdx0: Union[Sized, Iterable, Type]
    as_list: list

    @property
    def CurrencySet(self):
        return [c.name for c in self._CurrencyEnumIdx0]

    def __init__(self, _CurrencyEnumIdx0: Union[Sized, Iterable, Type]):
        self._CurrencyEnumIdx0 = _CurrencyEnumIdx0
        self.as_list = [0.] * (len(_CurrencyEnumIdx0))

    def unrealized(self, rates: Rates):
        result_home = 0.
        for Derived_CurrencyEnumIdx0, quantity in zip(self._CurrencyEnumIdx0, self.as_list):
            result_home += quantity * rates.of_currency(Derived_CurrencyEnumIdx0)
        return result_home

    def __getitem__(self, Derived_CurrencyEnumIdx0: Derived_CurrencyEnumIdx0):
        return self.as_list[Derived_CurrencyEnumIdx0.value]

    def __setitem__(self, Derived_CurrencyEnumIdx0: Derived_CurrencyEnumIdx0, value):
        self.as_list[Derived_CurrencyEnumIdx0.value] = value


@dataclass
class Direction:
    base_currency: Derived_CurrencyEnumIdx0
    quote_currency: Derived_CurrencyEnumIdx0
    home_currency: Derived_CurrencyEnumIdx0

    # this section for calculating swap
    swap_rate: float = 0.
    last_update_days = 0
    """
    Let:
        * I_XB = daily interest rate to borrow X from MM
        * I_XL = daily interest rate to lend X to MM
        * r_X = conversion rate with respect to the home Derived_CurrencyEnumIdx0 
    
    per day: SWAP = (I_t?*q_t*r_t/r_b + I_b?*q_b) amount of quote is gained:
    ? = B or L depending on the lending nature borrowing nature of the client toward MM.
    
    In buying position, q_t(t)>0, q_b(t)<0
    q_b(t+1) = q_b(t) * [1 - {SWAP}] = q_b(t) * [1 - { - I_tL*(q_t/q_b)*r_t/r_b - I_bB}]
    = q_b(t) * [1 - {I_tL - I_bB}] Since we know (q_t/q_b)*r_t/r_b ~ -1.
    The higher the SWAP = I_tL - I_bB, the better. Debt will be shrink with rate SWAP
    
    In selling position, q_t(t)<0, q_b(t)>0
    q_b(t+1) = q_b(t) * [1 + {SWAP}] = q_b(t) * [1 {I_tB*(q_t/q_b)*r_t/r_b + I_bL}]
    = q_b(t) * [1 + {I_bL - I_tB}] Since we know (q_t/q_b)*r_t/r_b ~ -1.
    The higher the SWAP = I_bL - I_tB, the better! MM has to pay you interest 
    
    the higher SWAP[A/B,buying or selling] the more advantageous (if the currencies are static) the nature (buy/sell).
    in reality, SWAP[A/B,buying] + SWAP[A/B,selling] < 0 coz MM needs to make money.
    """

    realized_home: float = 0.
    base_quantity: float = 0.
    quote_quantity: float = 0.
    isBuy: bool = True

    last_position_cached = math.nan

    @property
    def start_position(self):
        return -self.base_quantity / self.quote_quantity

    def open(self, lotsize_base: float, rates: Rates, daypassed=0):
        sign = +1 if self.isBuy else -1
        self.quote_quantity *= (1 - sign * self.swap_rate * daypassed)
        self.base_quantity += sign * lotsize_base
        self.quote_quantity -= sign * rates.of_currency(self.base_currency) / rates.of_currency(
            self.quote_currency) * lotsize_base

        self.last_position_cached = self.start_position

    def unrealized(self, rates: Rates):
        result = 0.
        result += self.base_quantity * rates.of_currency(self.base_currency)
        result += self.quote_quantity * rates.of_currency(self.base_currency)
        result /= rates.of_currency(self.home_currency)
        return result

    def close(self, lotsize_base: float, rates: Rates, daypassed=0):
        self.open(-lotsize_base, rates)
        quote_retained = -rates.of_currency(self.base_currency) / rates.of_currency(
            self.quote_currency) * self.base_quantity
        redeemed_quote = self.quote_quantity - quote_retained
        self.realized_home += redeemed_quote * rates.of_currency(self.quote_currency)
        self.quote_quantity -= redeemed_quote


@dataclass
class DeadlockAssets:
    Currency_type: Union[Sized, Iterable, Type]
    directions: List[Direction]

    def __init__(self, Currency_type: Union[Sized, Iterable, Type]):
        self.Currency_type = Currency_type
        self.directions = []

    def currency_basket(self):
        cb = CurrencyBasket(self.Currency_type)
        for position in self.directions:
            cb[position.base_currency] += position.base_quantity
            cb[position.quote_currency] += position.quote_quantity
        return cb

    def unrealized(self, rates: Rates):
        return self.currency_basket().unrealized(rates)

    @property
    def realized(self):
        return sum(position.realized_home for position in self.directions)

    @staticmethod
    def auto_add_into_basket(position_factory: Callable[[Any], Direction]):
        added_idx = -1

        def decorated(self) -> Direction:
            nonlocal added_idx
            self: DeadlockAssets
            if added_idx == -1:
                position = position_factory(self)
                added_idx = len(self.directions)
                self.directions.append(position)
            return self.directions[added_idx]

        return decorated

        # No setter coz position once defined is final
        # def decorated_setter(self, value):
        #     nonlocal added_at
        #     self: PositionAssets
        #     if added_at == -1:
        #         position = position_Creator(self)
        #         added_at = len(self.directions)
        #         self.directions.append(position)
        #     self.directions[added_at] = value
        # return property(decorated, fset=decorated_setter)
