# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import logging
import pandas as pd
import random


def get_mds(sample_path, mask_type=None):
    """Returns the voltage vs. time from a formatted .csv file."""
    df = pd.read_csv(sample_path)
    df = __shift_time(df, -30)
    return df


def __shift_time(df, time=-30):
    """Returns a data frame with the time values shifted by x seconds.
    """
    df['Seconds'] = df['Seconds']+time
    return df
