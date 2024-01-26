import pandas as pd  
from ML_Pipeline import utils
from ML_Pipeline import feature_engineering
from ML_Pipeline import train_model
features_df = pd.read_csv('D:/NIMESHA_P/Projects_2024/Zillow-House-Price-Prediction/Input/properties_2016.csv')
target_df = pd.read_csv('D:/NIMESHA_P/Projects_2024/Zillow-House-Price-Prediction/Input/train_2016_v2.csv')

initial_dataset = utils.merge_dataset(features_df,target_df,'parcelid')
print("Shape of Dataset: ",initial_dataset.shape)

utils.save_dataset(initial_dataset)

print('### Perform Feature Engineering ###')

feature_engineering.perform_feature_engineering(initial_dataset)
final_df =pd.read_csv('D:/NIMESHA_P/Projects_2024/Zillow-House-Price-Prediction/Input/final_zillow_dataset.csv')
model_path,test_df, test_labels  = train_model.train_and_save_model(final_df)

train_model.model_prediction(model_path,test_df,test_labels)