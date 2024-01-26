def merge_dataset(features_df, target_df, join_key):

    final_df = features_df.copy()
    final_df = final_df.merge(target_df,how = 'inner' , on =join_key)
    return final_df

def save_dataset(dataframe):

    dataframe.to_csv('D:/NIMESHA_P/Projects_2024/Zillow-House-Price-Prediction/Input/zillow_initial_dataset.csv',index=False)
    

