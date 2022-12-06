import sklearn as sk
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd



df_raw_data = pd.read_csv("CWPT2/trainDataset.csv")
df_test_data = pd.read_csv("CWPT2/testDatasetExample.csv")
nan_data = df_raw_data.replace(999, np.nan)
print(nan_data.isna().sum().sum())
