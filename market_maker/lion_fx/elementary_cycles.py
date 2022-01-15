import numpy as np


def main():
    id_of = {}

    def o(ab):
        a, b = ab
        return min(a, b), max(a, b)

    with open('elementary_cycles.in', 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [eval(l) for l in lines]
        lines = [l[-1] for l in lines]
        lines.sort(key=lambda x: len(x))
        for l in lines:
            for dir in zip(l, l[1:] + [l[0]]):
                if o(dir) not in id_of:
                    id_of[o(dir)] = len(id_of)

        basis = []
        elementaries = []
        for l in lines:
            y = np.zeros(len(id_of))
            for dir in zip(l, l[1:] + [l[0]]):
                y[id_of[o(dir)]] += 1. if o(dir) == dir else -1.
            if np.linalg.matrix_rank(basis + [y]) > np.linalg.matrix_rank(basis):
                basis.append(y)
                elementaries.append(l)
    print(id_of)
    print(lines)
    print(elementaries)


elementary_cycles = [['NZD', 'CHF', 'JPY'], ['NZD', 'GBP', 'CAD'], ['USD', 'CHF', 'CAD'], ['CAD', 'JPY', 'USD'],
                     ['NZD', 'GBP', 'CHF'], ['JPY', 'CHF', 'EUR'], ['CAD', 'GBP', 'USD'], ['CAD', 'JPY', 'CHF'],
                     ['CAD', 'GBP', 'CHF'], ['AUD', 'GBP', 'CAD'], ['USD', 'EUR', 'CHF'], ['CAD', 'JPY', 'AUD'],
                     ['NZD', 'GBP', 'AUD'], ['CAD', 'EUR', 'AUD'], ['AUD', 'CHF', 'JPY'], ['CHF', 'AUD', 'USD'],
                     ['GBP', 'CHF', 'JPY'], ['CAD', 'EUR', 'GBP'], ['EUR', 'GBP', 'USD'], ['EUR', 'JPY', 'NZD'],
                     ['CAD', 'JPY', 'NZD', 'USD'], ['MXN', 'USD', 'CHF', 'JPY']]

if __name__ == '__main__':
    main()
