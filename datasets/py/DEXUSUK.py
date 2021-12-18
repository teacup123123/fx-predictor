from datasets.general import *

gbp_in_usd_daily = readpickle(__file__.split('.')[0])
"""
Source: Board of Governors of the Federal Reserve System (US)  Release: H.10 Foreign Exchange Rates  
Units:  U.S. Dollars to One U.K. Pound Sterling, Not Seasonally Adjusted

Frequency:  Daily

Noon buying rates in New York City for cable transfers payable in foreign currencies.

Suggested Citation:
Board of Governors of the Federal Reserve System (US), U.S. Dollars to U.K. Pound Sterling Spot Exchange Rate [DEXUSUK], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXUSUK, December 18, 2021.
"""