import sklearn as sk
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


df_raw_data = pd.read_csv("trainDataset.csv")
df_test_data = pd.read_csv("testDatasetExample.csv")

cleanData(df_raw_data)

def cleanData(df_raw):
    print(df_raw.duplicated().sum())
    
