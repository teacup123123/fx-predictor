from datasets.general import *

usd_in_cad_daily = readpickle(__file__.split('.')[0])
"""
Source: Board of Governors of the Federal Reserve System (US)  Release: H.10 Foreign Exchange Rates  
Units:  Canadian Dollars to One U.S. Dollar, Not Seasonally Adjusted

Frequency:  Daily

Noon buying rates in New York City for cable transfers payable in foreign currencies.

Suggested Citation:
Board of Governors of the Federal Reserve System (US), Canadian Dollars to U.S. Dollar Spot Exchange Rate [DEXCAUS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DEXCAUS, December 18, 2021.
"""