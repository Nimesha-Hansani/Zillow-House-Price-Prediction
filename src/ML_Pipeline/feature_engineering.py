import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings 
from scipy import stats

pd.set_option('display.max_columns',None)
warnings.simplefilter(action='ignore')

def perform_feature_engineering(df):

    df.drop_duplicates(subset="parcelid",keep='first',inplace=True)
    print("Duplicates Dropped")

    ### Missing Values Handling

    mis_var = [var for var in df.columns if df[var].isnull().sum() >0]
    limit = np.abs((df.shape[0] * 0.6))
    var_to_be_dropped = [var for var in mis_var if df[var].isnull().sum() > limit]
    df.drop(columns=var_to_be_dropped, axis=1,inplace=True)
    print("Column with more missing values are dropped")

    
    ### Capture elapsed time
    df['yeardifference']  = df['assessmentyear']  - df['yearbuilt']
    print("Capture Elapsed Time")

    df.drop(columns=['assessmentyear','yearbuilt','transactiondate'],axis=1,inplace= True)
    print("Dropping all the time related columns")

    ### Transform incorrectly scaled variables
    df[['latitude','longitude']] =(df[['latitude','longitude']])/(10**6)
    df['censustractandblock'] = (df['censustractandblock'])/(10**12)
    df['rawcensustractandblock'] = (df['rawcensustractandblock'])/(10**6)   
    print("Transform incorrectly scaled variables")


    ### Handle missing values
    mis_var = [var for var in df.columns if df[var].isnull().sum() > 0]
    for var in mis_var:
        df[var] = df[var].fillna(df[var].mode()[0])
    print("Missing values handled")

    ### Encooding categorical variables
    catg_vars = [var for var in df.columns if df[var].dtypes == 'O']
    for i in range(len(catg_vars)):
 
        var = catg_vars[i]
        var_le = LabelEncoder()
        var_labels  = var_le.fit_transform(df[var])
        var_mappings = {index: label for index, label in enumerate(var_le.classes_)}
        df[(var + '_labels')] = var_labels
        df.drop(columns=var ,axis=1 ,inplace = True)

    print("Categorical variables encoded")

    ###Checking & Removing outliers

    z = np.abs(stats.zscore(df))
    no_out_df = df[(z<3).all(axis=1)]
    print(no_out_df.shape)
    print("Outliers Removed")

    ### Muliti Colinearity
    no_out_df.drop(columns=['calculatedbathnbr','calculatedfinishedsquarefeet','structuretaxvaluedollarcnt'], axis=1 ,inplace= True)
    no_out_df.drop(columns=['taxvaluedollarcnt','landtaxvaluedollarcnt','fullbathcnt'], axis=1 ,inplace= True)
    no_out_df.drop(columns=['censustractandblock','propertycountylandusecode_labels'],axis=1,inplace=True)
    print("High correlated data was dropped")


    no_out_df.to_csv('D:/NIMESHA_P/Projects_2024/Zillow-House-Price-Prediction/Input/final_zillow_dataset.csv',index=False)
    print("Pre-processed dataset saved")
