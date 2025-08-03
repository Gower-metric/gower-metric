import numpy as np
import pandas as pd
from gower_metric import Gower

# load data
iris = pd.read_csv("your_path_to_data/iris.csv")

dat1 = iris.iloc[0:10].reset_index(drop=True)
dat2 = iris.iloc[5:15].reset_index(drop=True)

# define feature types
f_types = {"sepal_length": "numeric", "sepal_width": "numeric", "petal_length": "numeric", "petal_width": "numeric", "variety": "categorical_nominal",}

# combine two sets for range scaling
union = pd.concat([dat1, dat2], ignore_index=True)

# call Gower
gower = Gower(feature_types = f_types, scale = "range").fit(union)
dist_py = np.array([gower(dat1.iloc[i], dat2.iloc[i]) for i in range(10)])

# print the results
print(dist_py)