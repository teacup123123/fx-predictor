from datasets.general import *

nzd_in_usd_daily = readpickle(__file__.split('.')[0])
"""
Source: Board of Governors of the Federal Reserve System (US)  Release: H.10 Foreign Exchange Rates  
Units:  U.S. Dollars to One New Zealand Dollar, Not Seasonally Adjusted

Frequency:  Daily

Noon buying rates in New York City for cable transfers payable in foreign currencies.

Suggested Citation:
Board of Governors of the Federal Reserve System (US), U.S. Dollars to New Zealand Dollar Spot Exchange Rate [DEXUSNZ], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXUSNZ, December 18, 2021.
"""
