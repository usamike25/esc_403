import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
from globals import MODELS, MANDATORY_PARAMETERS, LEVEL_1_PARAMS, LEVEL_2_PARAMS, ALL_PARAMS, TO_PREDICT

df = pd.read_excel('/Users/miguelmeier/Desktop/UZH/semester4/ESC_403/project/Synthetic_big.xlsx')
df.drop(['Unnamed: 0'], axis=1, inplace=True)

#INPUT --> PARAMETER LIST
def determine_best_model(columns_list=ALL_PARAMS):
    target_column = TO_PREDICT[0] 

    X = df[columns_list]  
    y = df[target_column] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test) 
        rmse = np.sqrt(mean_squared_error(y_test, predictions))  
        r2 = r2_score(y_test, predictions)  

        results[name] = {
            "R^2": r2,
            "RMSE": rmse,
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions,
            "model": model,
        }

        # plot_prediction_error(X_test, y_test, model, name)
        # plot_residuals(X_test, y_test, predictions, name)

    return results


def plot_prediction_error(X_test, y_test, predictions, model_name):

    # Prediction Error Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.3, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Line for perfect predictions
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title(f'Prediction Error for {model_name}')
    
    plt.savefig(f"prediction_error_{model_name}.png")
    plt.close()

def plot_residuals(X_test, y_test, predictions, model_name):
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig(f'Residual Plot for {model_name}')
    plt.close()


results = determine_best_model()

print(results)

