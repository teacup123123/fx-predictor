from dataclasses import dataclass

import datasets.yahoo as yh

# jpy = yh.load('JPY=X')
from basket import PositionAssets, Position, Currency, Rates, CurrencyBasket


@dataclass
class PA_eur_cad_jpy(PositionAssets):
    @PositionAssets.decorator_position
    def cad_jpy_b(self):
        return Position(Currency.cad, Currency.jpy, Currency.jpy, isBuy=True)

    @PositionAssets.decorator_position
    def eur_jpy_s(self):
        return Position(Currency.eur, Currency.jpy, Currency.jpy, isBuy=False)

    @PositionAssets.decorator_position
    def eur_cad_b(self):
        return Position(Currency.eur, Currency.cad, Currency.jpy, isBuy=True)


if __name__ == '__main__':
    rates = Rates()

    paloop = PA_eur_cad_jpy()
    ls = rates[Currency.eur] / rates[Currency.cad]
    paloop.cad_jpy_b.enter(ls, rates)
    paloop.eur_cad_b.enter(1, rates)
    paloop.eur_jpy_s.enter(1, rates)
    print(paloop)

    val = paloop.unrealized(rates)

    rates[Currency.cad] *= 1.1

    val = paloop.unrealized(rates)
    paloop.cad_jpy_b.exit(ls, rates)
    # paloop.eur_cad_b.exit(1, rates)
    # paloop.eur_jpy_s.exit(1, rates)
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
