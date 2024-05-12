import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from globals import MODELS, MANDATORY_PARAMETERS, LEVEL_1_PARAMS, LEVEL_2_PARAMS, ALL_PARAMS, TO_PREDICT, LEVEL_1_PARAMS_OPTIONAL

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
PURPLE = '\033[35m'
RESET = '\033[0m'

def load_data(file_name):
    return pd.read_csv(file_name)


def run_model(df, target, test_size=0.2):

    X = df.drop(target, axis=1)

    y = df[target]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)



    model = LinearRegression()

    model.fit(X_train_scaled, y_train)



    y_pred_train = model.predict(X_train_scaled)

    y_pred_test = model.predict(X_test_scaled)



    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    r2_train = r2_score(y_train, y_pred_train)

    r2_test = r2_score(y_test, y_pred_test)



    return X_train_scaled, X_test_scaled, y_train, y_test, y_pred_train, y_pred_test, rmse_train, rmse_test, r2_train, r2_test



def set_session_state():
    if 'level_1_computation_done' not in st.session_state:
        st.session_state.level_1_computation_done = False
    if 'level_2_computation_done' not in st.session_state:
        st.session_state.level_2_computation_done = False



# Streamlit UI

st.title('Fiber-Reinforced Concrete: Predicting Failure on Beam')



# File uploader

file = st.file_uploader('Choose a CSV file', type='csv')



if file is not None:

    df = load_data(file)

    set_session_state()

    st.write(df.head())

    st.divider()

    st.subheader("Parameter Choice")

    mandatory_param_string = ""

    for i, param in enumerate(MANDATORY_PARAMETERS):
        mandatory_param_string += param
        if i != len(MANDATORY_PARAMETERS) - 1:
            mandatory_param_string += ", "

    st.write(f"The parameters {mandatory_param_string} are mandatory, and thus always part of the prediction process.")

    options_level_1_params = st.multiselect("Select additional optional the parameters with which you'd like to predict the IFSS.", ["None"] + LEVEL_1_PARAMS_OPTIONAL)

    # level_1_par = MA
    if st.button("Predict IFSS"):
        st.session_state.level_1_computation_done = True


    st.divider()

    # Column selector

    target = st.selectbox('Select target variable', df.columns)
    # Model execution
    if st.session_state.level_1_computation_done:
        if st.button('Run Model'):

                X_train_scaled, X_test_scaled, y_train, y_test, y_pred_train, y_pred_test, rmse_train, rmse_test, r2_train, r2_test = run_model(df, target)



                st.write(f"RMSE Train: {rmse_train}")

                st.write(f"RMSE Test: {rmse_test}")

                st.write(f"R^2 Train: {r2_train}")

                st.write(f"R^2 Test: {r2_test}")



                # Visualization for model fit

                fig, axs = plt.subplots(1, 2, figsize=(15, 5))


                # Plot for training data

                axs[0].scatter(X_train_scaled[:,0], y_train, color="blue", label="Actual", alpha=0.5)

                axs[0].scatter(X_train_scaled[:,0], y_pred_train, color="red", label="Predicted", alpha=0.5)

                axs[0].set_title('Model Fit - Training Data')

                axs[0].set_xlabel('Features')

                axs[0].set_ylabel(target)

                axs[0].legend()


                # Plot for test data

                axs[1].scatter(X_test_scaled[:,0], y_test, color="blue", label="Actual", alpha=0.5)

                axs[1].scatter(X_test_scaled[:,0], y_pred_test, color="red", label="Predicted", alpha=0.5)

                axs[1].set_title('Model Fit - Test Data')

                axs[1].set_xlabel('Features')

                axs[1].set_ylabel(target)

                axs[1].legend()


                st.pyplot(fig)
