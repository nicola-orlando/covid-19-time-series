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

import pickle

# Used to merge different regions of some contries (Canada, France, ..)
def group_and_sum(dataframe,first_feature,second_feature,manipulated_data_first,manipulated_data_second,title_first,title_second): 
    grouped_data = dataframe.groupby([first_feature,second_feature])[manipulated_data_first,manipulated_data_second].sum().reset_index()
    return grouped_data

# Get list of countries to analyse
def list_of_countries(dataframe,test_run=True):
    print('\n----> Obtaining list of countries')
    countries=[]
    if test_run:
            countries.append('Italy')
    else: 
        countries=dataframe.Country_Region.unique()
    return countries 

# Plot input data 
def plot_series(dataframe,country):
    print('\n----> Plotting input series')
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

# Cleanup, rearrange and select a given country in the dataset
def prepare_dataset_for_training(dataframe,country,feature_to_predict_first,feature_to_predict_second,feature_to_predict_name_first,feature_to_predict_name_second):
    print('\n----> Minimal manipulation on training data')
    dataframe=dataframe[dataframe.Country_Region == country]    
    # Remove entries with no data
    dataframe=dataframe[feature_to_predict_first != 0]    
    dataframe=group_and_sum(dataframe,'Date','Country_Region',feature_to_predict_name_first,feature_to_predict_name_second,feature_to_predict_name_first,feature_to_predict_name_second) 
    return dataframe 

# Pickup the relevant country only
def prepare_dataset_for_testing(dataframe,country):
    print('\n----> Pickup the country from the testing data corresponding to the country selected in the training data')
    dataframe=dataframe[dataframe.Country_Region == country]    
    return dataframe 

# Same idea as illustrated in this post https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
def transpose_the_dataset(dataframe,feature,number_of_shifts=0):
    # Gets the full lenght of the dataset 
    print('\n----> Shifting the dataset to convert it into a useful format for a classification problem')
    if number_of_shifts == 0: 
        number_of_shifts = len(dataframe[feature])
    print('Transposing with '+str(number_of_shifts)+' data steps')
    print(dataframe.head())
    for index in range(1,number_of_shifts+1): 
        name_feature_added=feature+'_'+str(index)
        dataframe[name_feature_added]=dataframe[feature].shift(index)
    # The created dataset will have plenty of NaNs, for this specific case associated to days with no registed cases, get rid of them 
    dataframe = dataframe.fillna(0)
    output_df_and_shift = [dataframe,number_of_shifts]
    return output_df_and_shift

# For saving the model
def save_model(name,model):
  pickle.dump(model, open(name, 'wb'))

# Define and train the model, return the score
def train_model(X_train,y_train,output_modelname,print_verbose=False):
    print("\n----> Training a Bayesian Ridge model")
    bayesian_model = linear_model.BayesianRidge()
    bayesian_model.fit(X_train,y_train)
    score_predictions_train = bayesian_model.predict(X_train)
    if print_verbose: 
        print('Print score_predictions_train')
        print(score_predictions_train)
    save_model(output_modelname+'.pkl',bayesian_model)
    return score_predictions_train

# Prepare some validation plots for the model predictions on trained data
def plot_model_predictions_vs_time(dataframe,base_feature,model_prediction,base_feature_title): 
    print('\n----> Plotting model prediction vs time')
    x_len=len(dataframe[base_feature].to_numpy())*0.19
    dataframe['Data-Over-Predicted']=dataframe[base_feature]/dataframe[model_prediction]
    figure(figsize=(x_len, 4.8))
    ax=plt.axes()
    plt.plot(dataframe['Date'], dataframe[base_feature], marker='', linewidth=1, alpha=0.9, label=base_feature_title)
    plt.plot(dataframe['Date'], dataframe[model_prediction], marker='', linewidth=1, alpha=0.9, label='Predicted')
    plt.legend()
    plt.title(base_feature_title+' compared to model predictions')
    plt.grid(True)
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

#  Prepare some validation plots for the model predictions on trained data (2D plots)
def plot_data_vs_predicted(dataframe,base_feature,model_prediction,base_feature_title): 
    print('\n----> Plotting data vs prediction')
    plt.plot(dataframe[base_feature], dataframe[model_prediction], marker='', linewidth=1, alpha=0.9, label=base_feature_title,)
    plt.grid(True)
    plt.title(base_feature_title+' compared to model predictions')
    plt.xlabel('Data')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plot_name=country+'_data_vs_pred.png'
    plt.savefig(plot_name)
    plt.close()

# Add score on days which overalp in training and testing dataset
def add_score_on_overlapping_days(dataframe_base,dataframe_test):
    print('\n----> Add prediction on overlapping testing/training datasets')
    dataframe_merged = pd.merge(dataframe_base,dataframe_test,on='Date',how='left')
    # Ensure to keep only the intersection of the two sets by forcing the ID of the second to be not a Nan
    dataframe_merged=dataframe_merged[dataframe_merged.ForecastId==dataframe_merged.ForecastId]
    return dataframe_merged

# Obtain the first row in the testing dataset which has no prediction from the trained model, this is to be used to evaluate the model
# Expected output structure [ConfirmedCases_1,ConfirmedCases_75]
def extract_last_row_for_testing(dataframe,number_of_shifts,print_verbose=False): 
    print('Extract first emtpy row from testing data to be filled up with model prediction')
    if print_verbose: 
        print('In extract_last_row_for_testing printing input data')
        print(dataframe.head())
    # Extract first row without a prediction
    row_to_predict = dataframe[dataframe.score_predictions_bayes != dataframe.score_predictions_bayes].head(1)
    # Extract last row with a prediction, to be used to fillup the previous based on actual data and model prediction
    row_predicted = dataframe[dataframe.score_predictions_bayes == dataframe.score_predictions_bayes].tail(1)
    row_predicted_copy = row_predicted.copy()
    for index in range(1,number_of_shifts+1): 
        index_shifted = index-1 
        feature_original='ConfirmedCases'+'_'+str(index_shifted)
        feature_mod='ConfirmedCases'+'_'+str(index)
        if index == 1: 
            row_predicted['ConfirmedCases_1'].iloc[0]=row_predicted_copy['score_predictions_bayes'].iloc[0]
        else: 
            row_predicted[feature_mod].iloc[0]=row_predicted_copy[feature_original].iloc[0]
    if print_verbose:
        print('Original data with prediction')
        print(row_predicted_copy.head())
    # More output cleanup
    features_to_drop_output=['score_predictions_bayes']
    for feature_to_drop_output in features_to_drop_output:
        row_predicted = row_predicted.drop(feature_to_drop_output,1)
    if print_verbose:
        print('Manipulated data to be predicted')
        print(row_predicted.head())
    return row_predicted

def load_and_apply_the_model(dataframe,model_name,print_verbose=False):
    print('\n----> Loading and applying the model')
    loaded_model = pickle.load(open(model_name+'.pkl', 'rb'))
    score_prediction = loaded_model.predict(dataframe)
    if print_verbose:
        print('Predicted score '+str(score_prediction))
    return score_prediction

def replace_first_row(dataframe_bottom,row,print_verbose=False): 
    print('\n----> Replacing now the first row of the final testing dataset')
    if print_verbose:
        print(dataframe_bottom.head())
        print(row.head())
    row_array = row.to_numpy()
    dataframe_bottom.iloc[0]=row_array.squeeze()
    if print_verbose:
        print('After conversion')
        print(dataframe_bottom.head())
    return dataframe_bottom

# Header will look like this 
# Id Province_State Country_Region Date  ConfirmedCases  Fatalities
print("Load the datasets ...")
file_path_train="/afs/cern.ch/user/o/orlando/.keras/datasets/covid19-global-forecasting-week-4/train.csv"
file_path_test="/afs/cern.ch/user/o/orlando/.keras/datasets/covid19-global-forecasting-week-4/test.csv"
df_input_train=pd.read_csv(file_path_train)
df_input_test=pd.read_csv(file_path_test)
print("Looking at the raw input train data ...")
print(df_input_train.head(3))
print("Looking at the raw input test data ...")
print(df_input_test.head(3))

# Simple replacement(s)
df_input_train = df_input_train.replace({'Country_Region': {'"Korea, South"': 'Korea, South'}}) 
df_input_test = df_input_test.replace({'Country_Region': {'"Korea, South"': 'Korea, South'}}) 
# Get the countries we want to use
countries=list_of_countries(df_input_train)

for country in countries:
    
    # Clean up the dataset with info we don't necessarily need     
    df_for_training = prepare_dataset_for_training(df_input_train,country,df_input_train.ConfirmedCases,
                                                   df_input_train.Fatalities,'ConfirmedCases','Fatalities')
    # Just select the country
    df_for_testing = prepare_dataset_for_testing(df_input_test,country)
                                                  
    # This is unecessary and confusing, to be fixed..
    df_for_training_copy = df_for_training

    # Make plot of input series
    plot_series(df_for_training,country)
    # Arrange the dataset as needed for a classification problem 
    df_for_training=transpose_the_dataset(df_for_training,'ConfirmedCases')
    # Get the number of shifts and the actual output dataframe 
    number_of_shifts = df_for_training[1]
    df_for_training = df_for_training[0]

    # The created dataset will have plenty of NaNs, for this specific case associated to days with no registed cases, get rid of them 
    df_for_training = df_for_training.fillna(0)

    # Remove data we don't want to feed into the training 
    features_to_drop=['Date', 'Country_Region', 'Fatalities']
    for feature_to_drop in features_to_drop:
        df_for_training = df_for_training.drop(feature_to_drop,1)
    y_train = df_for_training.pop('ConfirmedCases')
    
    print('\nTraining dataset after arrangements needed to convert into a classification problem and cleanup')
    print(df_for_training.head())

    # Train the model and return the predictions for the training dataset
    scores = train_model(df_for_training,y_train,'logreg')

    # Append the scores to the training data, eventually need to clean this up as not all is necessary..
    dataframe_with_scores = pd.DataFrame(data = scores, columns = ['score_predictions_bayes'], index = df_for_training_copy.index.copy())
    output_dataframe = pd.merge(df_for_training_copy, dataframe_with_scores, how = 'left', left_index = True, right_index = True)
    print('\nPriting training data merged with the model predictions (output_dataframe)')
    print(output_dataframe.head())

    # Perform some plots
    plot_model_predictions_vs_time(output_dataframe,'ConfirmedCases','score_predictions_bayes','Confirmed cases')
    plot_data_vs_predicted(output_dataframe,'ConfirmedCases','score_predictions_bayes','Confirmed cases')

    print('\nNow looking at the dataset for testing for the first time (df_for_testing)')
    print(df_for_testing.head())
    dataframe_merged =  add_score_on_overlapping_days(df_for_testing,output_dataframe)
    
    # Basic cleanup operations on the testing dataframe which has been merged with the training set
    features_to_drop=['ConfirmedCases', 'Data-Over-Predicted', 'Date', 'Country_Region_x', 'Country_Region_y', 'Fatalities', 'ForecastId', 'Province_State']
    for feature_to_drop in features_to_drop:
        dataframe_merged = dataframe_merged.drop(feature_to_drop,1)

    print('\nPrint dataframe_merged (testing with scores appened) after cleanup')
    print(dataframe_merged.head())

    dataframe_to_be_evaluated = extract_last_row_for_testing(dataframe_merged,number_of_shifts)
    print('\nHere printing the one row dataframe holding the data to be used to obtain the prediction')
    print(dataframe_to_be_evaluated)
    
    score_to_be_saved = load_and_apply_the_model(dataframe_to_be_evaluated,'logreg')
    dataframe_to_be_evaluated['score_predictions_bayes']=score_to_be_saved
    print('\nOne row dataframe score appended')   
    print(dataframe_to_be_evaluated.head())
    
    #dataframe_merged = dataframe_merged.mask(dataframe_merged.score_predictions_bayes != dataframe_merged.score_predictions_bayes, dataframe_to_be_evaluated)
    print(dataframe_merged.head(15))

    dataframe_merged_first = dataframe_merged[dataframe_merged['score_predictions_bayes'] == dataframe_merged['score_predictions_bayes']]

    dataframe_merged_second = dataframe_merged[dataframe_merged['score_predictions_bayes'] != dataframe_merged['score_predictions_bayes']]
        
    dataframe_merged_second_replaced = replace_first_row(dataframe_merged_second,dataframe_to_be_evaluated)

    # Stick the data 
    dataframe_concatenated = pd.concat([dataframe_merged_first,dataframe_merged_second_replaced])
    print(dataframe_concatenated.head(15))
