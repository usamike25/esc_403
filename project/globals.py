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

MODELS = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForestRegressor": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    # "RandomForestClassi,fier": RandomForestClassifier(),
    "SVR": SVR(),
}

MANDATORY_PARAMETERS = ["expected concrete compression strength 95% rH [MPa]", "fiber tenacity [MPa]", "avg .circumference [mm]"]

ALL_PARAMS = ["Titer [tex]", "expected concrete compression strength 95% rH [MPa]", "avg .circumference [mm]", "fiber length [mm]", "W_q [µm]", "W_Sm [µm]", "IFSS [MPa]", "fiber tenacity [MPa]"]  

LEVEL_1_PARAMS = ["Titer [tex]", "expected concrete compression strength 95% rH [MPa]", "avg .circumference [mm]", "fiber length [mm]", "W_q [µm]", "W_Sm [µm]"]

LEVEL_1_PARAMS_OPTIONAL = ["Titer [tex]", "fiber length [mm]", "W_q [µm]", "W_Sm [µm]"]

LEVEL_1_PARAMS_MANDATORY = ["expected concrete compression strength 95% rH [MPa]", "avg .circumference [mm]"]

LEVEL_2_PARAMS = ["IFSS [MPa]", "fiber tenacity [MPa]"]

TO_PREDICT = ["IFSS [MPa]", "Minibeam energy absorption (m.e.a.) 0-11 [J]"]