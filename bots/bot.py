import dataclasses
from fx_library import *


@dataclasses.dataclass
class Position:
    base: CurrencyEnumIdx0
    quote: CurrencyEnumIdx0
    lot_unit: int
    lot_number: Union[float, int]
    nature: Nature

    swap_accumulated: float
    price_opened: float
