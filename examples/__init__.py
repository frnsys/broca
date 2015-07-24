import json
from os import getcwd, path

base_path = path.realpath(path.join(getcwd(), path.dirname(__file__)))
def load_data(filename):
    datapath = path.join(base_path, 'data', filename)
    with open(datapath, 'r') as f:
        data = json.load(f)
    return data
