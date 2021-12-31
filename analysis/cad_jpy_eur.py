from dataclasses import dataclass

import datasets.yahoo as yh

# jpy = yh.load('JPY=X')
from fx_library import DeadlockAssets, Direction, Currency, Rates, CurrencyBasket


@dataclass
class DL_eur_cad_jpy(DeadlockAssets):
    @property
    @DeadlockAssets.auto_add_into_basket
    def cad_jpy_b(self) -> Direction:
        return Direction(Currency.cad, Currency.jpy, Currency.jpy, isBuy=True)

    @property
    @DeadlockAssets.auto_add_into_basket
    def eur_jpy_s(self) -> Direction:
        return Direction(Currency.eur, Currency.jpy, Currency.jpy, isBuy=False)

    @property
    @DeadlockAssets.auto_add_into_basket
    def eur_cad_b(self) -> Direction:
        return Direction(Currency.eur, Currency.cad, Currency.jpy, isBuy=True)


if __name__ == '__main__':
    rates = Rates()

    paloop = DL_eur_cad_jpy()
    value_jpy = rates[Currency.eur] * 123456.546
    paloop.cad_jpy_b.open(value_jpy / rates[Currency.cad], rates)
    paloop.eur_cad_b.open(value_jpy / rates[Currency.eur], rates)
    paloop.eur_jpy_s.open(value_jpy / rates[Currency.eur], rates)
    print(paloop)

    val = paloop.unrealized(rates)

    rates[Currency.cad] *= 1.1

    val = paloop.unrealized(rates)
    paloop.cad_jpy_b.close(paloop.cad_jpy_b.base_quantity, rates)
    # paloop.eur_cad_b.close(1, rates)
    # paloop.eur_jpy_s.close(1, rates)
    cb = paloop.currency_basket()

    print()

    # print(rates.of_currency(Currency.cad))
    #
    # asset = CurrencyBasket()
    # asset.cad += 1000
    # asset.eur += 2000
    # print(asset.value_jpy(rates))
    #
    # positionAssets = PositionAssets()
    # print(positionAssets)

    # for time in times:
    #     pass
