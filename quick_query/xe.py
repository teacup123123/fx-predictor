import requests
import json

HEADERS = {}

sec_header = [
    {
        "name": "sec-ch-ua",
        "value": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"96\", \"Google Chrome\";v=\"96\""
    },
    {
        "name": "authorization",
        "value": "Basic bG9kZXN0YXI6djdhOFdUZHZ3MTRmV2hRUEJMTEdiam5VYTNEWGN5RmM="
    },
    {
        "name": "Referer",
        "value": "https://www.xe.com/currencycharts/?from=EUR&to=ILS&view=2Y"
    },
    {
        "name": "sec-ch-ua-mobile",
        "value": "?0"
    },
    {
        "name": "User-Agent",
        "value": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    },
    {
        "name": "sec-ch-ua-platform",
        "value": "\"Windows\""
    }
]
for _ in sec_header:
    HEADERS[_['name']] = _['value']


def grab(fr, to, long=False):
    if long:
        url = f'''https://www.xe.com/api/protected/charting-rates/?fromCurrency={fr}&toCurrency={to}&isExtended=true'''
        raise NotImplementedError
    else:
        url = f'''https://www.xe.com/api/protected/live-currency-rates/?currencyPairs={fr}%2F{to}'''
        got = requests.get(url, headers=HEADERS)
        content = got.content
        content, *_ = json.loads(content)
        return content['rate']


if __name__ == '__main__':
    print(grab('JPY', 'USD'))
