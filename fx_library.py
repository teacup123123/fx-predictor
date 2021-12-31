import math
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass, fields, field
from typing import List, Callable, Any


class EnumIdx0(Enum):
    """automatic enum structure that starts with index = 0 """

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
        self.as_list = [math.nan] * (len(Currency))
        self.home_currency = Currency.jpy
        self[Currency.jpy] = 1.
        self[Currency.eur] = 130.
        self[Currency.cad] = 90.
        self[Currency.usd] = 115.16

    def of_currency(self, c: Currency):
        return self.as_list[c.value]

    def __getitem__(self, currency: Currency):
        return self.as_list[currency.value]

    def __setitem__(self, currency: Currency, value):
        self.as_list[currency.value] = value


@dataclass
class CurrencyBasket:
    """A basket with different currencies of different interest rates"""
    as_list = [0.] * (len(Currency))

    def unrealized(self, rates: Rates):
        result_home = 0.
        for currency, quantity in zip(Currency, self.as_list):
            result_home += quantity * rates.of_currency(currency)
        return result_home

    def __getitem__(self, currency: Currency):
        return self.as_list[currency.value]

    def __setitem__(self, currency: Currency, value):
        self.as_list[currency.value] = value

    def __repr__(self):
        return f'CurrencyBasket{repr({k: v for k, v in zip(Currency, self.as_list)})}'


@dataclass
class Direction:
    base_currency: Currency
    quote_currency: Currency
    home_currency: Currency

    # this section for calculating swap
    swap_rate: float = 0.
    last_update_days = 0
    """
    Let:
        * I_XB = daily interest rate to borrow X from MM
        * I_XL = daily interest rate to lend X to MM
        * r_X = conversion rate with respect to the home currency 
    
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
    positions: List[Direction] = field(default_factory=list)

    def currency_basket(self):
        cb = CurrencyBasket()
        for position in self.positions:
            cb[position.base_currency] += position.base_quantity
            cb[position.quote_currency] += position.quote_quantity
        return cb

    def unrealized(self, rates: Rates):
        return self.currency_basket().unrealized(rates)

    @property
    def realized(self):
        return sum(position.realized_home for position in self.positions)

    @staticmethod
    def auto_add_into_basket(position_factory: Callable[[Any], Direction]):
        added_idx = -1

        def decorated(self) -> Direction:
            nonlocal added_idx
            self: DeadlockAssets
            if added_idx == -1:
                position = position_factory(self)
                added_idx = len(self.positions)
                self.positions.append(position)
            return self.positions[added_idx]

        return decorated

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


if __name__ == '__main__':
    # Here is a small example of usage
    rates = Rates()
    print(rates.of_currency(Currency.usd))
    print(rates)
    print(rates[Currency.usd])
    rates[Currency.usd] = 123.1453  # updates
    basket = CurrencyBasket()
    basket[Currency.usd] += 1.
    print(basket)
