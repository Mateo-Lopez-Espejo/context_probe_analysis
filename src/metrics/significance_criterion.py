import collections as col
import itertools as itt
import pathlib as pl
from configparser import ConfigParser
import joblib as jl

import numpy as np

import src.data.rasters
from src.data import LDA as cLDA, dPCA as cdPCA
from src.metrics import dprime as cDP
from src.data.load import load, get_site_ids
from src.data.cache import make_cache, get_cache, set_name
from src.metrics.reliability import signal_reliability
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))
site = 'CRD004a'
sites = [site]