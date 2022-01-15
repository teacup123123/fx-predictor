import os.path

import analysis
from analysis.parameters import AnalysisParameters, DEFAULT
import pickle as pk

_, OUTPUT = os.path.split(__file__)
OUTPUT = OUTPUT.replace('.py', '.pickle')


def loadData(params: AnalysisParameters = None):
    if params is None:
        params = DEFAULT
    if not os.path.exists(os.path.join(analysis.pickles_root, OUTPUT)):
        params.doStep0LoadData = True
    if params.p0LoadData.dataset == 'fred':
        import datasets.fred.unified_format as data_source

        if params.doStep0LoadData:
            data_source.initialize(params.currencies)
            asdict = data_source.as_TSS.as_dict()

            with open(os.path.join(analysis.pickles_root, OUTPUT), 'wb') as f:
                pk.dump((params.p0LoadData, asdict), f)
    elif params.p0LoadData.dataset == 'yahoo':
        import datasets.yahoo.unified_format as data_source

        if params.doStep0LoadData:
            data_source.init_pickles('10y', '1d')
            asdict = data_source.as_TSS.as_dict()
            for x in asdict:
                if x not in params.currencies:
                    asdict[x].clear()
            for x in params.currencies:
                assert x in asdict

            with open(os.path.join(analysis.pickles_root, OUTPUT), 'wb') as f:
                pk.dump((params.p0LoadData, asdict), f)
    else:
        raise ValueError('unknown dataset')

    with open(os.path.join(analysis.pickles_root, OUTPUT), 'rb') as f:
        paramsL, asdict = pk.load(f)
        if params.p0LoadData != paramsL:
            raise Exception("The saved data doesn't seem to match the specifications...")
    data_source.as_TSS.load_dict(asdict)
    return paramsL, asdict, data_source


def main():
    DEFAULT.doStep0LoadData = True
    return loadData()


if __name__ == '__main__':
    print(main())
