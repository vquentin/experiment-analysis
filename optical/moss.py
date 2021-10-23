# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import logging
import pandas as pd
import random


def get_mds_vs_t(sample_path, mask_type=None):
    """Returns the voltage vs. time from a formatted .csv file."""
    df = pd.read_excel(sample_path)
    return df
