import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import lightgbm as lgb
from datetime import datetime, timedelta
import matplotlib.dates as mdates