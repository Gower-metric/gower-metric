import pandas as pd
import numpy as np
import time

from gower_metric import Gower
from tqdm import tqdm

df = (pd.read_csv("your_path/adult_reduced.csv"))

f_types = {
    "age": "ratio_scale_interval",
    "education_num": "ratio_scale_interval",
    "race": "categorical_nominal",
    "sex": "categorical_nominal",
    "hours_per_week": "ratio_scale_interval",
}

gower = Gower(f_types, scale="range").fit(df)

N_EXPERIMENTS = 100
times = []

for _ in tqdm(range(N_EXPERIMENTS), desc="Cran Gower Python"):
    t0 = time.time()
    dist_py = np.array([gower(df.iloc[i], df.iloc[i+1]) for i in range(len(df)-1)])
    times.append(time.time() - t0)