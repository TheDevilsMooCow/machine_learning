import sklearn as sk
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

def dataPreprocessingClassification():
    df_raw_data = pd.read_excel('trainDataset.xls')
    test_data = pd.read_excel('testDatasetExample.xls')
    df_raw_data = df_raw_data.drop(['RelapseFreeSurvival (outcome)', 'ID'], axis = 1)
    test_data = test_data.drop(['ID'], axis = 1)
    #df_test_data = pd.read_excel('testDatasetExample.xls')

    clean, target = cleanData(df_raw_data)
    decision_tree_model = decisionTree(clean,target)
    #decision_tree_model.predict(test_data)
    return clean, target
    

def cleanData(df_raw):
    outlierCols = (df_raw.drop(['pCR (outcome)', 'ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage'], axis = 1)).columns

    #checks for duplicate values
    print(df_raw.duplicated().sum())
    print(df_raw)
    
    #converts the 999 values to NaN values due to functionality to remove NaN values
    df_raw = df_raw.replace(999, np.nan)
    print(df_raw.isna().sum().sum())
    
    #replaces NaN values with the medium of the column
    df_raw['TrippleNegative'] = df_raw['TrippleNegative'].fillna(df_raw['TrippleNegative'].mode()[0])
    df_raw['ChemoGrade'] = df_raw['ChemoGrade'].fillna(df_raw['ChemoGrade'].mode()[0])
    df_raw['Proliferation'] = df_raw['Proliferation'].fillna(df_raw['Proliferation'].mode()[0])
    df_raw['HistologyType'] = df_raw['HistologyType'].fillna(df_raw['HistologyType'].mode()[0])
    df_raw['LNStatus'] = df_raw['LNStatus'].fillna(df_raw['LNStatus'].mode()[0])
    df_raw['TumourStage'] = df_raw['TumourStage'].fillna(df_raw['TumourStage'].mode()[0])
    print(df_raw.isna().sum().sum())
    
    #drops any remaining NaN values
    df_raw = df_raw.dropna()
    print(df_raw.isna().sum().sum())
    
    #removes outliers
    #cols = list(df_raw.columns)

    print(df_raw.shape)
    df_raw = df_raw[(np.abs(stats.zscore(df_raw)) < 3).all(axis = 1)]
    
    #scales the data
    target = df_raw['pCR (outcome)']
    learning = df_raw.drop('pCR (outcome)', axis = 1)
    cols = list(learning.columns)
    scaler = StandardScaler()
    scaler.fit(learning)
    scaled = scaler.transform(learning)
    data = pd.DataFrame(scaled, columns=cols)
    
    #applies pca for optimal feature selection
    pca = PCA(.95)
    pca.fit(data)
    reduced = pca.transform(data)
    print(reduced.shape)
    return reduced, target
    

def decisionTree(X_train, y_train):
    clf = RandomForestClassifier(random_state=1, max_depth=10)
    dtt = clf.fit(X_train, y_train)
    return dtt
    

if __name__ == "__main__":
    dataPreprocessingClassification()