import os.path

_dir, _ = os.path.split(__file__)
with open(rf'{_dir}/currencies.txt', 'r') as _:
    currencies = _.read(-1).split()
