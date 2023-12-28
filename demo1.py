import streamlit as st
import pandas as pd
import plotly.express as px
from sarima import sarima_forecast


st.title('Data Visualization with Streamlit')


st.sidebar.header('Upload Data')
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

available_categories = []
min_date = None
max_date = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
  
   
    date_format = '%d-%m-%Y'  
    data['Order Date'] = pd.to_datetime(data['Order Date'], format=date_format, errors='coerce')

   
    date_column_name = 'Order Date'

    
    if date_column_name in data.columns:
        
        data[date_column_name] = pd.to_datetime(data[date_column_name], format='%d-%m-%Y', errors='coerce')

        
        min_date = data[date_column_name].min()
        max_date = data[date_column_name].max()

    if min_date is not None and max_date is not None:
        st.subheader(f"Data available for the date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    else:
        st.warning(f"No date column with the name '{date_column_name}' found in the dataset or unable to determine the date range.")

    available_categories = data.groupby('Category')['Sales'].sum().reset_index()

    st.subheader('Available Categories:')
    st.write(available_categories)
    
    st.subheader('Bar Chart of Sales by Category')
    bar_chart = px.bar(data, x='Category', y='Sales', title='Sales by Category')
    st.plotly_chart(bar_chart)

    
    category_sales = data.groupby('Category')['Sales'].sum()
    most_sold_category = category_sales.idxmax()
    least_sold_category = category_sales.idxmin()

    
    most_sold_sales = category_sales.max()
    least_sold_sales = category_sales.min()

    
    st.subheader('Most Sold Category:')
    st.write(f"The most sold category is {most_sold_category} with a total sales of {most_sold_sales}.")

    st.subheader('Least Sold Category:')
    st.write(f"The least sold category is {least_sold_category} with a total sales of {least_sold_sales}.")

if st.sidebar.button("Prediction") and data is not None:
    
    st.subheader('SARIMA Forecasting Results')
    sarima_models, forecasts = sarima_forecast(data)
else:
    st.info('Please upload a CSV file to see data visualizations.')
