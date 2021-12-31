from dataclasses import dataclass

import datasets.yahoo as yh

from fx_library import DeadlockAssets, Direction, CurrencyEnumIdx0, Rates


class Currency3(CurrencyEnumIdx0):
    jpy = ()
    cad = ()
    eur = ()


class Rates3(Rates):
    def __init__(self):
        super(Rates3, self).__init__(Currency3, Currency3.jpy)
        self[Currency3.jpy] = 1.
        self[Currency3.eur] = 130.
        self[Currency3.cad] = 90.


@dataclass
class DL_eur_cad_jpy(DeadlockAssets):
    def __init__(self): super(DL_eur_cad_jpy, self).__init__(Currency3)

    @property
    @DeadlockAssets.auto_add_into_basket
    def cad_jpy_b(self) -> Direction:
        return Direction(Currency3.cad, Currency3.jpy, Currency3.jpy, isBuy=True)

    @property
    @DeadlockAssets.auto_add_into_basket
    def eur_jpy_s(self) -> Direction:
        return Direction(Currency3.eur, Currency3.jpy, Currency3.jpy, isBuy=False)

    @property
    @DeadlockAssets.auto_add_into_basket
    def eur_cad_b(self) -> Direction:
        return Direction(Currency3.eur, Currency3.cad, Currency3.jpy, isBuy=True)


if __name__ == '__main__':
    rates = Rates3()

    paloop = DL_eur_cad_jpy()
    value_jpy = rates[Currency3.eur] * 123456.546
    paloop.cad_jpy_b.open(value_jpy / rates[Currency3.cad], rates)
    paloop.eur_cad_b.open(value_jpy / rates[Currency3.eur], rates)
    paloop.eur_jpy_s.open(value_jpy / rates[Currency3.eur], rates)
    print(paloop)

    val = paloop.unrealized(rates)

    rates[Currency3.cad] *= 1.1

    val = paloop.unrealized(rates)
    paloop.cad_jpy_b.close(paloop.cad_jpy_b.base_quantity, rates)
    # paloop.eur_cad_b.close(1, rates)
    # paloop.eur_jpy_s.close(1, rates)
    cb = paloop.currency_basket()
    print(cb)
    print(cb.unrealized(rates))
