import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

def sarima_forecast(data, forecasting_period=12):
    
    selected_columns = ["Order Date", "Sales", "Category"]
    data = data[selected_columns]

   
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    data = data.dropna(subset=['Order Date'])

    
    unique_categories = data['Category'].unique()

 
    sarima_models = {}
    forecasts = {}

    for category in unique_categories:
        
        category_data = data[data['Category'] == category]

       
        category_data['Order Date'] = pd.to_datetime(category_data['Order Date'])
        category_data.set_index("Order Date", inplace=True)

       
        monthly_data = category_data['Sales'].resample('M').sum()

        result = adfuller(monthly_data)

        if result[1] <= 0.05:
            d = 0
        else:
            d = 0
            while result[1] > 0.05:
                d += 1
                monthly_data = monthly_data.diff().dropna()
                result = adfuller(monthly_data)

       
        train_data = monthly_data.iloc[:-forecasting_period]
        test_data = monthly_data.iloc[-forecasting_period:]

       
        p_values = range(3)
        q_values = range(3)
        P_values = range(3)
        D_values = range(5)
        Q_values = range(3)
        S = 12

        best_mae = float('inf')
        best_params = None

        for p in p_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                
                                model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, S))
                                model_fit = model.fit()

                               
                                forecast = model_fit.get_forecast(steps=forecasting_period, dynamic=True)
                                mae = mean_absolute_error(test_data, forecast.predicted_mean)

                                if mae < best_mae:
                                    best_mae = mae
                                    best_params = (p, d, q, P, D, Q)
                            except Exception as e:
                                continue

        if best_params is None:
            print(f"No valid parameters found for category: {category}. Skipping.")
            continue

        
        p, d, q, P, D, Q = best_params
        final_model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, S))
        final_model_fit = final_model.fit()

       
        sarima_models[category] = final_model_fit

        
        forecast = final_model_fit.get_forecast(steps=forecasting_period, dynamic=True)
        forecasts[category] = forecast.predicted_mean

        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name="Training Data"))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name="Test Data", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast.predicted_mean.index, y=forecast.predicted_mean, mode='lines', name="Predicted Data", line=dict(color='red')))
        fig.update_layout(title=f"Category: {category} - Training vs. Test vs. Predicted Data", xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig)


        forecast_next_year = final_model_fit.get_forecast(steps=24, dynamic=True)

       
        fig_future_prediction = go.Figure()
        fig_future_prediction.add_trace(go.Scatter(x=forecast_next_year.predicted_mean.index[12:], y=forecast_next_year.predicted_mean[12:], mode='lines', name="Next Year Forecast", line=dict(color='green')))
        fig_future_prediction.update_layout(title=f"Category: {category} :>Future Prediction", xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig_future_prediction)



    return sarima_models, forecasts
