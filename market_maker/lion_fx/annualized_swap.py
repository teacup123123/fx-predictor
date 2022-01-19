import os.path
import pickle
from typing import Tuple, List, Dict

from market_maker import lion_fx


def loadSwap() -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    with open(os.path.join(lion_fx.root, 'graph.pickle'), 'rb') as f:
        nodes, links = pickle.load(f)
    return nodes, links


if __name__ == '__main__':
    loadSwap()
