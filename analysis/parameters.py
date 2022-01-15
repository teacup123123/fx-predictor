import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class Step0Params:
    currencies_space_delimited_str: str
    dataset: str


@dataclasses.dataclass
class Step1Params:
    days = 365 * 10


@dataclasses.dataclass
class Step2CParams:
    doOrigininal: bool = False
    hipassOriginal: int = 365 * 2

    doHighPassed: bool = True


@dataclasses.dataclass
class Step2OParams:
    PROFIT_RANGES: np.ndarray = np.arange(0.2, 5.0, 0.1)
    percentage: float = 0.95


@dataclasses.dataclass
class AnalysisParameters:
    doStep0LoadData: bool
    p0LoadData: Step0Params

    doStep1LogPercentRemoveTrend: bool
    p1LogPercentRemoveTrend: Step1Params

    doStep2Correlation: bool = False
    p2Correlation: Step2CParams = Step2CParams()

    doStep2COptimizeProfitRange: bool = False
    p2OptimizeProfitRange: Step2OParams = Step2OParams()

    @property
    def currencies(self) -> List[str]:
        return self.p0LoadData.currencies_space_delimited_str.upper().split()


DEFAULT = AnalysisParameters(
    doStep0LoadData=False,
    p0LoadData=Step0Params(
        currencies_space_delimited_str='EUR CHF JPY NZD GBP USD CAD mxn aud zar',
        dataset='yahoo'
    ),
    doStep1LogPercentRemoveTrend=False,
    p1LogPercentRemoveTrend=Step1Params()
)
