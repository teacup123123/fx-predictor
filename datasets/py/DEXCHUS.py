from datasets.general import *

usd_in_rmb_daily = readpickle(__file__.split('.')[0])
"""
Units:  Chinese Yuan Renminbi to One U.S. Dollar, Not Seasonally Adjusted

Frequency:  Daily

Noon buying rates in New York City for cable transfers payable in foreign currencies.

Suggested Citation:
Board of Governors of the Federal Reserve System (US), Chinese Yuan Renminbi to U.S. Dollar Spot Exchange Rate [DEXCHUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXCHUS, December 18, 2021.
"""