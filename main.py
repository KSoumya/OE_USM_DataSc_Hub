import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly 
import pylab

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

st.markdown(
	"""
	<style>
	.main{
	background-color: #F5F5F5;
	}
	</style>
	""",
	unsafe_allow_html = True
	)

@st.cache
def get_data(filename):
	df=pd.read_csv(filename)
	return df

with header:
	st.title('Welcome to my awesome DSc project!')
	st.text('This project shows DO classification at iosbaths 10m and 20m')

with dataset:
	st.header('POC July Cruise - DO sensor data')
	st.text('The DO sensor is a prototype sensor on the Triton vehicle')

	df = get_data("data/DO_sensor_20m.csv")
	df_fil = df[(df['DEPTH (m)'] > 11) | (df['DEPTH (m)'] < 8)] 
	df_fil['isobath'] = df['DEPTH (m)'].apply(lambda x: 10 if x <= 8 else 20)

	st.write(df_fil.head())

	st.subheader('Number of samples collected at each isobath')

	DO_cnt = pd.DataFrame(df_fil['isobath'].value_counts())
	st.bar_chart(DO_cnt)

	st.markdown("At 10m isobath, fewer samples were collected")

	# Create distplot with custom bin_size
	df_fil['isobath'] = pd.Categorical(df_fil['isobath'])

	df2 = df_fil
	df2['count']=range(df_fil.shape[0])
    
    # Plot!
	fig = plt.figure(figsize=(10, 14))
	sns.lineplot(x = "count", y = "O2 (mg/L)", hue = 'isobath', data = df2)
	st.pyplot(fig)


	
with features:
	st.header('Depth (m) and DO (mg/L)')
	st.text("We got only depth and DO (mg/L) info now, we'll upgrade later")

with modelTraining:
	st.header('RF and KNN')
	st.text("We'll see if RF and KNN classifiers have different performance!")

	sel_col, disp_col = st.columns(2)

	max_depth = sel_col.slider('Max depth of the model?', min_value = 5, max_value = 500, value = 20, step = 5)
	n_estimators = sel_col.selectbox('Max number of trees?', options = [100, 200, 300, "No Limit"], index = 0)
	input_feature = sel_col.text_input('Which feature should be used as the predictor?', 'DEPTH (m)')

	sel_col.text('List of features available in this data')
	sel_col.write(df2.columns)

	# Start Model Training
	if n_estimators == 'No Limit':
		regr = RandomForestRegressor(max_depth = max_depth)
	else:
		regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
	X = df2[['DEPTH (m)']]
	y = df2[['O2 (mg/L)']]

	regr.fit(X, y)
	prediction = regr.predict(y)

	disp_col.subheader("Mean Absolute Error (MAE) of the RF regressor is: ")
	disp_col.write(mean_absolute_error(y, prediction))

	disp_col.subheader("Mean Squared Error (MSE) of the RF regressor is: ")
	disp_col.write(mean_squared_error(y, prediction))

	disp_col.subheader("R squared score of the RF regressor is: ")
	disp_col.write(r2_score(y, prediction))