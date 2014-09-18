import json
from os import getcwd, path

__location__ = path.realpath(path.join(getcwd(), path.dirname(__file__)))
datapath = path.join(__location__, 'data/imdb_plot_sample.json')

with open(datapath, 'r') as data:
    docs = json.load(data)
