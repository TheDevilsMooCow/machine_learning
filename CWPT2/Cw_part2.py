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
    df_raw = df_raw.replace(999, np.nan)
    print(df_raw.isna().sum().sum())
    df_raw['TrippleNegative'] = df_raw['TrippleNegative'].fillna(df_raw['TrippleNegative'].mode()[0])
    df_raw['ChemoGrade'] = df_raw['ChemoGrade'].fillna(df_raw['ChemoGrade'].mode()[0])
    df_raw['Proliferation'] = df_raw['Proliferation'].fillna(df_raw['Proliferation'].mode()[0])
    df_raw['HistologyType'] = df_raw['HistologyType'].fillna(df_raw['HistologyType'].mode()[0])
    df_raw['LNStatus'] = df_raw['LNStatus'].fillna(df_raw['LNStatus'].mode()[0])
    df_raw['TumourStage'] = df_raw['TumourStage'].fillna(df_raw['TumourStage'].mode()[0])

    print(df_raw.isna().sum().sum())
    
if __name__ == "__main__":
    main()