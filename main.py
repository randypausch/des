import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import Holt as ExponentialSmoothing
import plotly.graph_objects as go
import csv
import datetime
from datetime import date


#Replace districts_datapath to the location containing districts.csv downloaded from covid19india.org
districts_csv_url = "https://api.covid19india.org/csv/latest/districts.csv"




def read_districts_csv(state="Tamil Nadu"):
	dataframe = pd.read_csv(districts_csv_url)
	tn = dataframe[dataframe.State == state]
	districts = tn.District
	district_ls = list(set(districts)) #Name of the districts
	district_ls = sorted(district_ls)
	districts_data = []
	results_for_csv = []
	for i in range(len(district_ls)):
		districts_data.append(tn[tn.District == district_ls[i]])
	return districts_data, district_ls

def des_model(data):

	model = ExponentialSmoothing(data[:-1],damped_trend=True).fit()
	sigma_hat = model.sse 
	current = data[-1]
	predictions = model.predict(start =0, end=len(data)+14).astype('int')
	act_init = len(data)
	max_value = max(data)+20
	for act in range(act_init,len(predictions)):
		data.append(None)
		data[act]=None
	
	parameters = model.params_formatted
	alpha = float(parameters.iloc[0].param)
	beta = float(parameters.iloc[1].param)
	phi = float(parameters.iloc[4].param)
	sigmas = [0 for i in range(len(predictions))]
	for j,h in zip(range(act_init-1,len(predictions)),range(16)):
		t1 = 1 + (np.square(alpha) * (h - 1))
		t2 = (beta * phi * h) / np.square(1 - phi)
		t3 = (2 * alpha * (1 - phi)) + (beta * phi)
		t4 = -beta * phi * (1 - np.power(phi, h))
		t5 = (np.square(1 - phi)) * (1 - np.square(phi))
		t6 = 2 * alpha * (1 - np.square(phi))
		t7 = beta * phi * (1 + (2 * phi) - np.power(phi, h))
		sig2_h = (sigma_hat/15 * ( t1 + (t2 * t3) + (t4/t5) * (t6 + t7) ))
		sigmas[j] = int(np.sqrt(sig2_h))


	return data, predictions, sigmas

def doubling_rate(confirmed, recovered, deceased):
	todays_growth = ((list(confirmed)[-1])-(list(recovered)[-1]))-(list(deceased)[-1])
	one_week_back = ((list(confirmed)[-8])-(list(recovered)[-8]))-(list(deceased)[-8])
	doubling_rate = 7 * np.log(2)/(np.log(todays_growth/one_week_back))
	if doubling_rate < 0:
		doubling_rate = abs(doubling_rate)
		return "Halving Rate", doubling_rate
	return "Doubling Rate", doubling_rate






if __name__ == '__main__':	
	districts_data, district_ls = read_districts_csv()
	figures = []
	start_date = date(2021,2,28)
	end_date = datetime.date.today() - datetime.timedelta(days=1)
	pred_end = end_date + datetime.timedelta(days=15) #Accounting for 0 in the DES Model :p

	diff_for_df = (start_date - end_date).days #Useful for comprehending

	for i in tqdm(range(len(district_ls))):
		if i != 0  and i != 15 and i != 22 and i != 38: #Ignoring Airport Quarantine, Railway Quarantine, Mayiladuthurai, and unknown
			fig = go.Figure()
			confirmed = districts_data[i].Confirmed
			recovered = districts_data[i].Recovered
			deceased = districts_data[i].Deceased
			strr, doubling = doubling_rate(confirmed, recovered, deceased)
      

			if i == 13: #Handling Krishnagiri's missing values 
				confirmed = list(confirmed)
				for _ in range(9):
					confirmed.insert(0,0) #Won't matter since we are anyways only doing for the second wave! :-)
				confirmed = np.array(confirmed)
				confirmed = pd.Series(confirmed)

			#Finding the diff between consecutive confirmed cases
			confirmed = confirmed.diff()
			confirmed = confirmed[1:] #First value is always None when using diff :/
			confirmed = confirmed[diff_for_df:] #Start only from March 1, as we are interested only in the second wave
			confirmed = confirmed.rolling(window=7).mean()#7 day moving average
			confirmed = np.absolute(confirmed)
			daily_new = list(confirmed.values.astype('int'))
			daily_new = daily_new[6:] #First 7 values will be None because of the MA :/
			
			daily_new, predictions, sigmas = des_model(daily_new)
			
			
			dates_for_predict = pd.date_range(start="2021-03-7",end=pred_end)
			
			#This line is for the spatial analysis. Please keep it commented if not using.

			# results_for_csv.append(["District: {}".format(str(district_ls[i])),"Actual New Cases on {}: {}".format(end_date.strftime("%d %B, %Y"),str(current)),"Mean Projected New Cases(14 days): {}".format(str(np.mean(predictions[act_init:-1]).astype('int'))),"{}: {:0.2f}".format(strr,doubling)])
			
		
			fig.add_trace(go.Scatter(x=dates_for_predict,y=daily_new,name="Actual New Cases ({})".format(end_date.strftime("%d %B, %Y")),visible=True))
			fig.add_trace(go.Scatter(x=dates_for_predict,y=predictions,error_y=dict(
		            type='data', # value of error bar given in data coordinates
		            array=sigmas,
		            visible=True),name="Predicted New Cases (DES)"))
			

				
			fig.update_traces(mode="markers+lines")
      
			fig.update_xaxes(showspikes=True)
			fig.update_yaxes(showspikes=True)
			fig.update_yaxes(rangemode="nonnegative")
			fig.update_layout(
			    title="{} - New Cases and Predictions using DES - {}: {}".format(str(district_ls[i]), strr, doubling),
          xaxis_title="Dates",
			    yaxis_title="New Cases"
			)
			figures.append(fig)
	with open('results.html', 'w') as f:
		for fig in figures:
			f.write(fig.to_html( include_plotlyjs='cdn'))
