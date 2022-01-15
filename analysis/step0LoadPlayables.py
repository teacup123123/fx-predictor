import os.path
import re
import project
import pickle as pk

from analysis.parameters import AnalysisParameters, DEFAULT


def loadPlayables(params: AnalysisParameters = None):
    if params is None:
        params = DEFAULT
    with open(os.path.join(project.root, r'market_maker\lion_fx\playable.txt'), 'r') as f:
        lines = f.readlines()
        groups = [re.match('(...)->(...):([+-.0-9]+)%', line).groups() for line in lines]
        playables = {
            (a, b): float(c) for a, b, c in groups
            if a in params.currencies and b in params.currencies
        }
    return playables

def main():
    return loadPlayables()

if __name__ == '__main__':
    print(main())
