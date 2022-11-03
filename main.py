import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly 
import pylab

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, plot_confusion_matrix, plot_precision_recall_curve

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

st.set_option('deprecation.showPyplotGlobalUse', False)

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
    ## Refer this link: https://seaborn.pydata.org/generated/seaborn.lineplot.html
	fig = plt.figure(figsize=(10, 14))
	sns.lineplot(x = "count", y = "O2 (mg/L)", hue = 'isobath', data = df2)
	st.pyplot(fig)


	
with features:
	st.header('Depth (m) and DO (mg/L)')
	st.markdown('* **first feature:** We got DO (mg/L)')
	st.markdown('* **second feature:** We got isobaths, 10 & 20m')

with modelTraining:
	st.header('RF and KNN')
	st.text("We'll see if RF and KNN classifiers have different performance!")

	sel_col, disp_col = st.columns(2)

	# Feature Transformation
	class_le = LabelEncoder()
	y = class_le.fit_transform(df2['isobath'].values)
	pd.value_counts(y)

	# Data Preparation
	X = df2['O2 (mg/L)']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
	X_train = X_train.values.reshape(X_train.shape[0],1)
	X_test = X_test.values.reshape(X_test.shape[0],1)

	n_neighbors = sel_col.slider('How many nearest neighbors?', min_value = 2, max_value = 10, value = 5, step = 1)

	# Model Training and Evaluation

    # KNN Modeling
	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)

	knn_report = pd.DataFrame(classification_report(y_true = y_test, 
		y_pred = y_pred, 
		output_dict=True)).transpose()

	disp_col.subheader("Accuracy of the KNN classifier is: ")
	disp_col.write(knn.score(X_test, y_test))
	st.subheader("Confusion Matrix of the KNN classifier is: ")
	plot_confusion_matrix(knn, X_test, y_test)
	st.pyplot()
	
	# RF Modeling
	max_depth = sel_col.slider('Max depth of the model?', min_value = 5, max_value = 500, value = 20, step = 5)
	n_estimators = sel_col.selectbox('Max number of trees?', options = [100, 200, 300, "No Limit"], index = 0)
	input_feature = sel_col.text_input('Which feature should be used as the predictor?', 'DEPTH (m)')

	sel_col.text('List of features available in this data')
	sel_col.write(df2.columns)

	if n_estimators == 'No Limit':
		rfm = RandomForestClassifier(max_depth = max_depth)
	else:
		rfm = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)

	rfm.fit(X_train, y_train)
	rf_pred = rfm.predict(X_test)
	
	rfm_report = pd.DataFrame(classification_report(y_true = y_test, 
		y_pred = rf_pred, 
		output_dict=True)).transpose()

	disp_col.subheader("Accuracy of the RF classifier is: ")
	disp_col.write(rfm.score(X_test, y_test))
	st.subheader("Confusion Matrix of the RF classifier is: ")
	plot_confusion_matrix(rfm, X_test, y_test)
	st.pyplot()

	knn_col, rf_col = st.columns(2)
	knn_col.subheader("Classification Report of the KNN model: ")
	knn_col.write(knn_report)
	rf_col.subheader("Classification Report of the RF model: ")
	rf_col.write(rfm_report)
