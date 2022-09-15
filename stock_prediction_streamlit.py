# load library
import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import yfinance as yf
import datetime
import plotly.express as px


# emojis = https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title = 'Stock Prediction',
                   page_icon=":bar_chart:",
                   layout="wide"
                   )

# print title of web app
st.title("주식 데이터 시각화 및 예측:bar_chart:")
st.markdown("> *Yahoo Finance*로부터 가져온 **주식 데이터**를 시각화 및 분석하고 미래 주가를 예측합니다.")
st.markdown("> 좌측 사이드 바를 이용하여 원하는 시작 날짜를 정할 수 있습니다.:calendar:")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load data from yahoo finance.
start = st.sidebar.date_input("Start date",dt.datetime(2017,1,1))
#end = st.sidebar.date_input('End date', datetime(2022,1,1))
# start=dt.date(2021,1,1)
end=dt.date.today()
#st.markdown("원하는 회사를 입력해주세요. *ex)apple=AAPL, google=GOOG, samsung=005930.KS*")
stocklist=('AAPL','GOOG','TSLA','MSFT','005930.KS','LPL','000660.KS')
st.markdown("```Apple = 'AAPL', Google = 'GOOG', Tesla = 'TSLA', Microsoft = 'MSFT', Samsung='005930.KS', LG Display = 'LPL', SK Hynix = '00060.KS'``` ")

option = st.selectbox('원하는 회사를 선택해주세요',stocklist)
#user_input=st.text_input('','AAPL')
data=pdr.get_data_yahoo(option,start,end)
#data=pdr.get_data_yahoo("005930.KS", start, end)

#fill nan vale with next value within columns
data.fillna(method="ffill",inplace=True)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')
st.markdown("")
# create checkbox
if st.checkbox('Show raw data'):
    st.subheader(f'Raw data of {option}')
    st.write(data)
 
# show the description of data
st.subheader(f'Detail description about Datasets : {option}')
descrb=data.describe()
st.write(descrb)

#create new columns like year, month, day
data["Year"]=data.index.year
data["Month"]=data.index.month
data["Weekday"]=data.index.day_name()

# dislay graph of open and close column
fig_close = px.line(data[['Open','High','Low','Close']])
st.subheader(f"'Open,High,Low,Close' Line Chart : {option}")
st.plotly_chart(fig_close)
# st.subheader('Graph of Close & Open:-')
# st.line_chart(data[['Open','High','Low','Close']])

# display plot of Adj Close column in datasets
st.subheader(f'Graph of Adjacent Close:{option}')
st.bar_chart(data['Adj Close'])

#High-low
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
data
st.subheader(f'HL_PCT, PCT_change Graph : {option}')
st.markdown("* HL_PCT = (High - Low)/ Close * 100")
st.markdown("* PCT_change = (Close - Open) / Open * 100")
fig_1=px.area(data[['HL_PCT','PCT_change']])
st.plotly_chart(fig_1)

# display plot of volume column in datasets
st.subheader(f'Graph of Volume : {option}')
st.bar_chart(data['Volume'])

# create new cloumn for data analysis.
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
data = data[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

# display the new dataset after modificaton
st.subheader(f'Newly format DataSet:{option}')
st.dataframe(data.tail(500))

forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.01 * len(data)))
data['labels'] = data[forecast_col].shift(-forecast_out)

X = np.array(data.drop(['labels'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
data.dropna(inplace=True)
y = np.array(data['labels'])

# split dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

# display the accuracy of forecast value.
st.subheader('Accuracy:')
st.write(confidence)

forecast_set = clf.predict(X_lately)
data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)]+[i]
    last_date = data.iloc[-1].name
    dti = pd.date_range(last_date, periods=forecast_out+1, freq='D')
    index = 1
for i in forecast_set:
    data.loc[dti[index]] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    index +=1

# display the forecast value.
st.subheader(f'Forecast value : {option} with LinearRegression')
st.dataframe(data.tail(50))

# display the graph of adj close and forecast columns
st.subheader(f'Graph of Adj Close and Forecast : {option}')
st.area_chart(data[["Adj Close","Forecast"]])