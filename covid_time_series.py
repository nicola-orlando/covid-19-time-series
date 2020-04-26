from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure

#from collections import Counter
import math

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

import pandas as pd

from sklearn import linear_model 
from sklearn.linear_model import BayesianRidge, LinearRegression

# Used to merge different regions of some contries (Canada, France, ..)
def group_and_sum(dataframe,first_feature,second_feature,manipulated_data_first,manipulated_data_second,title_first,title_second): 
    grouped_data = dataframe.groupby([first_feature,second_feature])[manipulated_data_first,manipulated_data_second].sum().reset_index()
    return grouped_data

def list_of_countries(dataframe,test_run=True):
    countries=[]
    if test_run:
            countries.append('Italy')
    else: 
        countries=dataframe.Country_Region.unique()
    return countries 

def plot_series(dataframe,country):
    features=['ConfirmedCases','Fatalities']
    dataframe=dataframe[dataframe.Country_Region == country]    
    dataframe=dataframe[dataframe.ConfirmedCases != 0]    
    dataframe['FatalitiesPerCases']=dataframe['Fatalities']/dataframe['ConfirmedCases']
    x_len=len(dataframe.Country_Region.to_numpy())*0.19
    # Cumulative cases
    dataframe.plot(x="Date", y=features, kind="bar",figsize=(x_len,4.8))
    ax=plt.axes()
    plt.xticks(rotation=90)
    plt.title('Confirmed cases and fatalities in '+country)
    plt.grid(True,axis='y')
    plt.tight_layout()
    plt.yscale('log') 
    plt.show()
    plot_name=country+'_evol.png'
    plt.savefig(plot_name)
    plt.close()
    # Cumulative fatalities per case
    dataframe=dataframe[dataframe.FatalitiesPerCases != 0]    
    dataframe.plot(x="Date", y='FatalitiesPerCases', kind="line",figsize=(x_len,4.8),marker='o')
    plt.title('Fatalities per confirmed cases in '+country)
    ax=plt.axes()
    plt.grid(True,axis='y')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_name=country+'_ratio.png'
    plt.savefig(plot_name)
    plt.close()

def prepare_dataset_for_training(dataframe,country,feature_to_predict_first,feature_to_predict_second,feature_to_predict_name_first,feature_to_predict_name_second):
    dataframe=dataframe[dataframe.Country_Region == country]    
    # Remove entries with no data
    dataframe=dataframe[feature_to_predict_first != 0]    
    dataframe=group_and_sum(dataframe,'Date','Country_Region',feature_to_predict_name_first,feature_to_predict_name_second,feature_to_predict_name_first,feature_to_predict_name_second) 
    return dataframe 

# Same idea as illustrated in this post https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
def transpose_the_dataset(dataframe,feature,number_of_shifts=0):
    # Gets the full lenght of the dataset 
    if number_of_shifts == 0: 
        number_of_shifts = len(dataframe[feature])
    print('Transposing with '+str(number_of_shifts)+' data steps')
    print(dataframe.head())
    for index in range(1,number_of_shifts+1): 
        name_feature_added=feature+'_'+str(index)
        dataframe[name_feature_added]=dataframe[feature].shift(index)
    # The created dataset will have plenty of NaNs, for this specific case associated to days with no registed cases, get rid of them 
    dataframe = dataframe.fillna(0)
    print(dataframe.head())
    return dataframe

def train_model(X_train,y_train,print_verbose=True):
    print("Training a Bayesian model")
    bayesian_model = linear_model.BayesianRidge()
    bayesian_model.fit(X_train,y_train)
    score_predictions_train = bayesian_model.predict(X_train)
    if print_verbose: 
        print('Print score_predictions_train')
        print(score_predictions_train)
    return score_predictions_train

def plot_model_predictions_vs_time(dataframe,base_feature,model_prediction,base_feature_title): 
    x_len=len(dataframe[base_feature].to_numpy())*0.19
    dataframe['Data-Over-Predicted']=dataframe[base_feature]/dataframe[model_prediction]
    figure(figsize=(x_len, 4.8))
    ax=plt.axes()
    plt.plot(dataframe['Date'], dataframe[base_feature], marker='', linewidth=1, alpha=0.9, label=base_feature_title)
    plt.plot(dataframe['Date'], dataframe[model_prediction], marker='', linewidth=1, alpha=0.9, label='Predicted')
    plt.legend()
    plt.title(base_feature_title+' compared to model predictions')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_name=country+'_pred_vs_time.png'
    plt.savefig(plot_name)
    plt.close()
    # Ratio data over predicted 
    ax=plt.axes()
    dataframe.plot(x="Date", y='Data-Over-Predicted', kind="line",figsize=(x_len,4.8),marker='o')
    plt.title(base_feature_title+' data over predicted '+country)
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_name=country+'_data_over_predicted.png'
    plt.savefig(plot_name)
    plt.close()

def plot_data_vs_predicted(dataframe,base_feature,model_prediction,base_feature_title): 
    plt.plot(dataframe[base_feature], dataframe[model_prediction], marker='', linewidth=1, alpha=0.9, label=base_feature_title,)
    plt.grid(True)
    plt.title(base_feature_title+' compared to model predictions')
    plt.xlabel('Data')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plot_name=country+'_data_vs_pred.png'
    plt.savefig(plot_name)
    plt.close()

# Header will look like this 
# Id Province_State Country_Region Date  ConfirmedCases  Fatalities
print("Load the dataset ...")
file_path_train="/afs/cern.ch/user/o/orlando/.keras/datasets/covid19-global-forecasting-week-4/train.csv"
file_path_test="/afs/cern.ch/user/o/orlando/.keras/datasets/covid19-global-forecasting-week-4/test.csv"
df_input_train=pd.read_csv(file_path_train)
df_input_test=pd.read_csv(file_path_train)
print("Looking at the raw input train data ...")
print(df_input_train.head())
print("Looking at the raw input test data ...")
print(df_input_test.head())

# Simple replacement(s)
df_input_train = df_input_train.replace({'Country_Region': {'"Korea, South"': 'Korea, South'}}) 
# Get the countries we want to use
countries=list_of_countries(df_input_train)

for country in countries:
    
    # Clean up the dataset with info we don't necessarily need     
    df_for_training = prepare_dataset_for_training(df_input_train,country,df_input_train.ConfirmedCases,
                                                   df_input_train.Fatalities,'ConfirmedCases','Fatalities')
    df_for_testing = prepare_dataset_for_training(df_input_train,country,df_input_train.ConfirmedCases,
                                                  df_input_train.Fatalities,'ConfirmedCases','Fatalities')
    df_for_training_copy = df_for_training
    
    # Make plot 
    #plot_series(df_for_training,country)
    # Arrange the dataset as needed for a classification problem 
    df_for_training=transpose_the_dataset(df_for_training,'ConfirmedCases')
    df_for_testing=transpose_the_dataset(df_for_training,'ConfirmedCases')
    # The created dataset will have plenty of NaNs, for this specific case associated to days with no registed cases, get rid of them 
    df_for_training = df_for_training.fillna(0)
    df_for_testing= df_for_testing.fillna(0)

    print(df_for_training.head())
    # Remove data we don't want to feed into the training 
    features_to_drop=['Date', 'Country_Region', 'Fatalities']
    for feature_to_drop in features_to_drop:
        df_for_training = df_for_training.drop(feature_to_drop,1)
        df_for_testing = df_for_testing.drop(feature_to_drop,1)
    y_train = df_for_training.pop('ConfirmedCases')
    y_test = df_for_testing.pop('ConfirmedCases')
    
    print(df_for_training.head())
    scores = train_model(df_for_training,y_train)

    dataframe_with_scores = pd.DataFrame(data = scores, columns = ['score_predictions_bayes'], index = df_for_training_copy.index.copy())
    output_dataframe = pd.merge(df_for_training_copy, dataframe_with_scores, how = 'left', left_index = True, right_index = True)
    print(output_dataframe.head())
    output_dataframe.to_csv('full_dataset_covid.csv', index=False) 

    plot_model_predictions_vs_time(output_dataframe,'ConfirmedCases','score_predictions_bayes','Confirmed cases')
    plot_data_vs_predicted(output_dataframe,'ConfirmedCases','score_predictions_bayes','Confirmed cases')
    
    # Compare Date series for training and testing data 
    print('\nTraining data\n')
    print(df_for_training.head())
    print('\nTesting data\n')
    print(df_for_testing.head())
