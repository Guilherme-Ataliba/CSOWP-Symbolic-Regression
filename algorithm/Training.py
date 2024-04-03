import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz-10.0.1-win64/bin/'

from functools import singledispatchmethod
from typing import Any, List
import pandas as pd
import numpy as np
from random import randint, choice, uniform
import graphviz
import matplotlib.pyplot as plt

from CSOWP_SR import *
from ExpressionTree import *

