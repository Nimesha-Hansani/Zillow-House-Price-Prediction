from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso  
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from math import sqrt  
import warnings 
warnings.simplefilter(action='ignore')
import pickle
import pandas as pd

def train_random_forest_reg(X_train, y_train, model_path):
    # param_grid = [
    # {'n_estimators': [300, 400, 500], 'max_features': [2,4,6]},
    # {'bootstrap':[False], 'n_estimators': [300, 400, 500], 'max_features': [2, 4, 6]}
    # ]

    param_grid = [
    {'n_estimators': [300], 'max_features': [2]},
    {'bootstrap':[False], 'n_estimators': [300], 'max_features': [2]}
    ]
    
    print("---Model Training Started---")
    forest_regressor = RandomForestRegressor()
    grid_search = GridSearchCV(forest_regressor, param_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=5 )
    grid_search.fit(X_train, y_train)
    print(" Finding best estimator")
    
    final_predictor = grid_search.best_estimator_
    final_predictor.fit(X_train, y_train)

    pickle.dump(final_predictor,open(model_path, 'wb'))

    return None



def train_and_save_model(df):

    X = df.drop('logerror',axis =1)
    y = df['logerror']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state= 100)

    train_vars = [ var for var in X_train.columns if var not in ['parcelid','logerror']]
    
    scaler = StandardScaler()
    scaler.fit(X_train[train_vars])
    X_train[train_vars] =scaler.transform(X_train[train_vars])
    X_test[train_vars] =scaler.transform(X_test[train_vars])

    X_train_new = X_train.copy()
    X_test_new = X_test.copy()

    X_train.drop(columns="parcelid", axis=1, inplace=True)
  
    model_path = "D:/NIMESHA_P/Projects_2024/Zillow-House-Price-Prediction/Output/HP_Model.pkl"
    
    train_random_forest_reg(X_train,y_train, model_path)

    return model_path, X_test, y_test
    
def model_prediction(model_path, X_test,y_test):

    X_test_new = X_test.copy()
    X_test.drop(columns="parcelid", axis=1, inplace=True)
    
    model = pickle.load(open(model_path,'rb'))
    predictions = model.predict(X_test)

    print('Mean Absolute Error : {}'.format(mean_absolute_error(y_test, predictions)))
    print()
    print('Mean Squared Error : {}'.format(mean_squared_error(y_test, predictions)))
    print()
    print('Root Mean Squared Error : {}'.format(sqrt(mean_squared_error(y_test, predictions))))
    print()

    predition_df = pd.DataFrame({'parcelid':X_test_new.parcelid, 'logerror': predictions})

    print(predition_df.head(5))

