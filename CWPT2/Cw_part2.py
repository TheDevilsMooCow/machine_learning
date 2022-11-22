import sklearn as sk
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df_raw_data = pd.read_excel('trainDataset.xls')
    df_test_data = pd.read_excel('testDatasetExample.xls')
    
    cleanData(df_raw_data)

def cleanData(df_raw):
    print(df_raw.duplicated().sum())
    df_raw = df_raw.replace(['999'], 'NaN')
    
    
if __name__ == "__main__":
    main()